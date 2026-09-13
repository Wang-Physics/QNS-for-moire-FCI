"""v4 outer-C3 sectors: shared mixture20 -> private100.

Two spawned GPU lanes keep parameters local. Every shared update samples all
three strata, evaluates every sector on the same 4128 configurations, solves
MBAR weights, then applies separate weighted SR updates from the same pre-update
pool. No inter-sector parameter averaging or seed/checkpoint selection occurs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE = {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now().astimezone().isoformat()


def gpu_env(gpu):
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               JAX_COMPILATION_CACHE_DIR=str(ROOT / '.jax-cache'),
               OMP_NUM_THREADS='16', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    return env


def initialize_lane(gpu, sectors, config):
    os.environ.update(gpu_env(gpu))
    os.environ.pop('LD_LIBRARY_PATH', None)
    os.sched_setaffinity(0, range(0, 16) if gpu == 0 else range(32, 48))
    import jax
    from .jax_neural_bloch import JaxNeuralBlochSpec, initialize, local_energy
    from .active_support import ActiveSupportController
    from .run_jax_neural_bloch import JaxMetropolis, parser, _save_checkpoint
    from .mixture_sr import WeightedNaturalGradient
    STATE['started'] = time.monotonic()
    STATE['config'] = config
    STATE['models'] = {}
    # Distinct strata of one reproducibly generated shared pool.
    rng = np.random.default_rng(config['seed'] + 1009)
    initial_x = rng.random((config['samples'], config['particles'], 2))
    initial_l = rng.integers(0, 2, (config['samples'], config['particles']), dtype=np.int32)
    count = config['samples'] // 3
    report = []
    for m in sectors:
        v4 = config['ansatz_version'] == 'v4'
        spec = JaxNeuralBlochSpec(n_particles=config['particles'], width=config['width'], message_passing_steps=2,
                                 cells=config['cells'], minor_chunk=config['minor_chunk'],
                                 backflow_init_scale=config['backflow_init_scale'],
                                 active_gamma_minors=config['active_gamma_minors'],
                                 determinants=1, orbital_hidden=config['width'],
                                 fixed_gamma_no_m=not v4,
                                 translation_projected_rank=1 if v4 else 0,
                                 outer_c3_projector=True,
                                 c3_irrep=m)
        params, constants = initialize(spec, config['seed'])
        optimizer = WeightedNaturalGradient(constants, spec, config['chunk'], 0.01)
        flat = optimizer.bind(params)
        sampler = JaxMetropolis(params, constants, spec, count, 0.06,
                                config['seed']+1009+m, config['wave_batch'])
        sampler.positions = jax.numpy.asarray(initial_x[m*count:(m+1)*count])
        sampler.layers = jax.numpy.asarray(initial_l[m*count:(m+1)*count])
        sampler.refresh(params)
        policy = parser().parse_args([])
        policy.cg_true_residual_interval = 5
        output = Path(config['output']) / 'shared20' / f'p{m}'
        metadata = dict(spec=asdict(spec), seed=config['seed'], sector=m,
                        training_step=0, phase='shared_mixture', protocol=config)
        _save_checkpoint(output/'jax_neural_bloch_initial.npz', flat, metadata, sampler)
        STATE['models'][m] = dict(
            parameters=params, flat=flat, constants=constants, spec=spec,
            optimizer=optimizer, sampler=sampler, policy=policy, output=output,
            recovery=False, stable=0, rejected=0, lr=0.002, damping=0.01,
            trace=[], metadata=metadata,
            energy_fn=jax.jit(lambda p, x, l, support, c=constants, s=spec:
                              local_energy(p,x,l,c,s,support)),
            support_controller=(ActiveSupportController(
                spec, constants, config['seed']+271828+m,
                candidate_pool=config['active_support_candidate_pool'],
                replace_fraction=config['active_support_replace_fraction'],
                score_walkers=config['active_support_score_walkers'],
            ) if config['active_gamma_minors'] is not None else None),
            support_trace=[],
        )
        report.append(dict(sector=m, parameter_sha256=hashlib.sha256(
            np.asarray(flat).tobytes()).hexdigest(), gpu=gpu, pid=os.getpid()))
    return report


def sample_lane(sweeps):
    results = []
    for m, model in STATE['models'].items():
        start = time.monotonic()
        a = model['sampler'].sweep(model['parameters'], sweeps)
        model['acceptance'] = a
        model['sample_seconds'] = time.monotonic()-start
        results.append(dict(sector=m, positions=np.asarray(model['sampler'].positions),
                            layers=np.asarray(model['sampler'].layers), acceptance=a,
                            seconds=time.monotonic()-start))
    return results


def evaluate_lane(positions, layers):
    import jax.numpy as jnp
    from .run_jax_neural_bloch import _batched
    x, l = jnp.asarray(positions), jnp.asarray(layers)
    result = []
    for m, model in STATE['models'].items():
        start = time.monotonic()
        logs = model['sampler'].evaluate(model['parameters'], x, l)
        energies = _batched(
            lambda p, bx, bl: model['energy_fn'](
                p, bx, bl, model['optimizer'].support_indices
            ), model['parameters'], x, l, STATE['config']['energy_batch']
        )
        model['pool'] = (x, l, energies)
        real_logs, energy = np.asarray(2*logs.real), np.asarray(energies)
        model['evaluation_seconds'] = time.monotonic()-start
        if not np.isfinite(real_logs).all() or not np.isfinite(energy).all():
            raise RuntimeError(f'nonfinite shared evaluation in P{m}; stopped')
        result.append(dict(sector=m, log_density=real_logs, energies=energy,
                           seconds=time.monotonic()-start))
    return result


def update_lane(weights, step, weight_diagnostics):
    import jax.numpy as jnp
    from .run_jax_neural_bloch import (
        _adaptive_update_is_accepted, _adaptive_next_state, _save_checkpoint,
    )
    result = []
    for m, model in STATE['models'].items():
        start = time.monotonic()
        x, l, energies = model['pool']
        w = weights[m]
        optimizer = model['optimizer']
        optimizer.set_weights(w)
        optimizer.damping = model['damping']
        policy = model['policy']
        direction, diagnostics = optimizer.direction(
            model['flat'], x, l, energies, policy.cg_max_iterations, 0.0,
            min_iterations=policy.cg_min_iterations, true_tolerance=0.02,
            true_residual_interval=5,
        )
        if not np.isfinite(np.asarray(direction)).all():
            raise RuntimeError(f'nonfinite SR direction P{m}')
        accepted = _adaptive_update_is_accepted(diagnostics, policy)
        energy = np.asarray(energies)
        mu = np.sum(w*energy)
        record = dict(
            sector=m, step=step, phase='shared_mixture', energy_meV=float(mu.real),
            energy_per_particle_meV=float(mu.real/model['spec'].n_particles),
            variance_meV2=float(np.sum(w*(energy.real-mu.real)**2)),
            importance_ess=float(1/np.sum(w*w)), update_accepted=accepted,
            learning_rate=model['lr'], sr_damping=model['damping'],
            **model['acceptance'], **diagnostics,
        )
        if accepted:
            next_flat = model['flat'] - model['lr']*direction
            if not np.isfinite(np.asarray(next_flat)).all():
                raise RuntimeError(f'nonfinite updated parameters P{m}')
            model['flat'] = next_flat
            model['parameters'] = optimizer.unravel(next_flat)
            model['sampler'].refresh(model['parameters'])
        else:
            model['rejected'] += 1
        support_record = None
        if (
            model['support_controller'] is not None
            and step <= STATE['config']['active_support_freeze_after']
            and step % STATE['config']['active_support_refresh_interval'] == 0
        ):
            model['parameters'], new_support, support_record = (
                model['support_controller'].update(
                    model['parameters'], optimizer.support_indices,
                    x, l, energies, step, weights=w,
                )
            )
            model['flat'] = optimizer.bind(model['parameters'])
            optimizer.set_support(new_support)
            model['sampler'].set_support(new_support, model['parameters'])
            support_record['rethermalization_acceptance'] = model['sampler'].sweep(
                model['parameters'],
                STATE['config']['active_support_rethermalize_sweeps'],
            )
            model['support_trace'].append(support_record)
            write_json(
                model['output']/'active_support_trace.json',
                model['support_trace'],
            )
        (model['recovery'], model['stable'], model['lr'], model['damping']) = (
            _adaptive_next_state(accepted, model['recovery'], model['stable'], policy))
        record.update(rejected_updates_cumulative=model['rejected'],
                      active_support_updated=support_record is not None,
                      active_support_update_seconds=(None if support_record is None
                                                     else support_record['seconds']),
                      sr_and_update_seconds=time.monotonic()-start,
                      step_seconds=(time.monotonic()-start+model['sample_seconds']+model['evaluation_seconds']),
                      next_learning_rate=model['lr'], next_sr_damping=model['damping'])
        model['trace'].append(record)
        state = dict(next_learning_rate=model['lr'], next_sr_damping=model['damping'],
                     recovery_mode=model['recovery'], recovery_stable_count=model['stable'],
                     rejected_updates=model['rejected'])
        metadata = dict(model['metadata'], training_step=step, optimizer_state=state,
                        mixture_diagnostics=weight_diagnostics, record=record)
        model['last_metadata'] = metadata
        _save_checkpoint(model['output']/'jax_neural_bloch_latest.npz', model['flat'],
                         metadata, model['sampler'])
        if step % 10 == 0 or step == STATE['config']['pretrain_steps']:
            _save_checkpoint(model['output']/f'jax_neural_bloch_step_{step:04d}.npz',
                             model['flat'], metadata, model['sampler'])
        write_json(model['output']/'jax_neural_bloch_trace.json', model['trace'])
        result.append(record)
    return result


def fork_lane(positions, layers):
    """Use precisely the last update, not the lowest observed energy."""
    import jax
    from types import SimpleNamespace
    from .run_jax_neural_bloch import _save_checkpoint
    output = Path(STATE['config']['output'])
    result = []
    for m, model in STATE['models'].items():
        key = jax.random.fold_in(jax.random.PRNGKey(STATE['config']['seed']+1009), 100+m)
        snapshot = SimpleNamespace(
            positions=positions, layers=layers, key=key,
            support_indices=model['optimizer'].support_indices,
        )
        path = output/f'p{m}'/'shared20_final_for_branch.npz'
        metadata = dict(model['last_metadata'], phase='fork_after_shared20',
                        fork_rule='last checkpoint, no best-energy selection',
                        branch_burn_sweeps=STATE['config']['branch_burn'],
                        full_pool_copied=True, independent_sampler_key=True)
        _save_checkpoint(path, model['flat'], metadata, snapshot)
        write_json(path.parent/'jax_neural_bloch_trace.json', model['trace'])
        write_json(path.parent/'active_support_trace.json', model['support_trace'])
        result.append(dict(sector=m, checkpoint=str(path),
                           parameter_sha256=hashlib.sha256(np.asarray(model['flat']).tobytes()).hexdigest(),
                           pool_sha256=hashlib.sha256(np.asarray(positions).tobytes()+np.asarray(layers).tobytes()).hexdigest(),
                           sampler_key=np.asarray(key).tolist()))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(forks=result, gpu_lane_wall_seconds=time.monotonic()-STATE['started'],
                cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime)


def collect(pools, function, *args):
    pending = [pool.submit(function, *args) for pool in pools]
    result = []
    for future in pending:
        result.extend(future.result())
    return sorted(result, key=lambda item: item['sector'])


def run_branch(config, gpu, m):
    output = Path(config['output'])/f'p{m}'
    command = ['/usr/bin/time', '-v', '-o', str(output/'resource_usage.txt'),
               'taskset', '-c', '0-15' if gpu == 0 else '32-47',
               str(ROOT/'.venv/bin/python'), '-m', 'src.run_jax_neural_bloch',
               '--output-dir', str(output), '--resume', str(output/'shared20_final_for_branch.npz'),
               '--particles', str(config['particles']),
               ('--v4-gamma-projected-m' if config['ansatz_version']=='v4'
                else '--fixed-gamma-no-m'), '--outer-c3-projector',
               '--cells', str(config['cells']), '--width', str(config['width']),
               '--backflow-init-scale', str(config['backflow_init_scale']),
               '--minor-chunk', str(config['minor_chunk']),
               '--wavefunction-batch', str(config['wave_batch']),
               '--local-energy-batch', str(config['energy_batch']),
               '--c3-irrep', str(m), '--seed', str(config['seed']),
               '--samples', str(config['samples']), '--steps', str(config['pretrain_steps']+config['branch_steps']),
               '--burn-sweeps', str(config['branch_burn']), '--sweeps-per-step', '2',
               '--sr-chunk', str(config['chunk']), '--cg-true-residual-interval', '5',
               '--checkpoint-interval', '10', '--log-interval', '10']
    if config['active_gamma_minors'] is not None:
        command += [
            '--active-gamma-minors', str(config['active_gamma_minors']),
            '--active-support-refresh-interval', str(config['active_support_refresh_interval']),
            '--active-support-replace-fraction', str(config['active_support_replace_fraction']),
            '--active-support-candidate-pool', str(config['active_support_candidate_pool']),
            '--active-support-score-walkers', str(config['active_support_score_walkers']),
            '--active-support-freeze-after', str(config['active_support_freeze_after']),
            '--active-support-rethermalize-sweeps', str(config['active_support_rethermalize_sweeps']),
        ]
    if config['smoke']:
        command += ['--smoke-test']
    command += ['--adaptive-cg-120-step']
    status = dict(status='running', sector=m, gpu=gpu, started=stamp(), command=command)
    start = time.monotonic()
    with (output/'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=gpu_env(gpu),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status['pid'] = process.pid
        write_json(output/'run_status.json', status)
        while process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            trace_path = output/'jax_neural_bloch_trace.json'
            try:
                trace = json.loads(trace_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if any(not np.isfinite(row['energy_per_particle_meV']) for row in trace):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                status['failure'] = 'nonfinite training energy'
                break
    final = output/'jax_neural_bloch_final.json'
    complete = False
    if process.returncode == 0 and final.exists() and 'failure' not in status:
        metadata = json.loads(final.read_text())
        trace = json.loads((output/'jax_neural_bloch_trace.json').read_text())
        complete = (metadata['training_step'] == config['pretrain_steps']+config['branch_steps']
                    and len(trace) == metadata['training_step']
                    and all(np.isfinite(row['energy_per_particle_meV']) for row in trace))
        if complete:
            tail = np.array([r['energy_per_particle_meV'] for r in trace[-min(10,config['branch_steps']):]])
            status.update(tail_mean_meV_per_particle=float(tail.mean()), tail_rms=float(tail.std()))
    status.update(status='complete' if complete else 'failed', exit_code=process.returncode,
                  finished=stamp(), wall_seconds=time.monotonic()-start)
    write_json(output/'run_status.json', status)
    print(json.dumps(status), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3184)
    parser.add_argument('--particles', type=int, choices=(3,6,9,18), default=6)
    parser.add_argument('--cells',type=int,choices=(9,27),default=9)
    parser.add_argument('--ansatz-version', choices=('v3','v4'), default='v4')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.particles not in (args.cells//3,2*args.cells//3):
        raise ValueError('particle number does not match cell count')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('refusing duplicate experiment launch')
    config = dict(output=str(output), seed=args.seed,
                  ansatz_version=args.ansatz_version,
                  ansatz_definition=(
                      f'P_m [(1/{args.cells}) sum_T det(M D_T Q)] with one dense M'
                      if args.ansatz_version=='v4' else
                      'legacy P_m applied to no-M+M_S'),
                  parameter_seeds=[args.seed]*3,
                  particles=args.particles, cells=args.cells, filling='1/3' if args.particles==args.cells//3 else '2/3',
                  width=64 if args.cells==27 else 32,
                  backflow_init_scale=0.01 if args.cells==27 else 1.0,
                  minor_chunk=256 if args.cells==27 else 128,
                  active_gamma_minors=(
                      256 if args.cells==27 and args.ansatz_version=='v3' else None
                  ),
                  active_support_refresh_interval=1 if args.smoke_test else 5,
                  active_support_replace_fraction=0.25,
                  active_support_candidate_pool=64 if args.smoke_test else 4096,
                  active_support_score_walkers=8 if args.smoke_test else 32,
                  active_support_freeze_after=2 if args.smoke_test else 60,
                  active_support_rethermalize_sweeps=2,
                  wave_batch=43 if args.cells==27 else 258,
                  energy_batch=1 if args.cells==27 else 32,
                  samples=24 if args.smoke_test else 4128,
                  chunk=(
                      8 if args.smoke_test else (43 if args.cells==27 else 258)
                  ),
                  pretrain_steps=1 if args.smoke_test else 20,
                  branch_steps=1 if args.smoke_test else 100,
                  burn=3 if args.smoke_test else 300,
                  branch_burn=3 if args.smoke_test else 300,
                  smoke=args.smoke_test,
                  objective='(E0(theta0)+E1(theta1)+E2(theta2))/3; each energy normalized',
                  parameters='same random initialization, independent arrays and updates',
                  shared_sampling='equal-size sector strata pooled; MBAR normalized importance weights',
                  metric='independent weighted sector SR; 1/3 cancels between force and metric; per-sector damping',
                  cg_true_residual_interval=5, checkpoint_selection='last pretraining update only',
                  source='Shirts and Chodera, JCP 129,124105 (2008), arXiv:0801.1426',
                  limitations='finite-sample estimated norms; persistent MCMC; weight ESS is not time-series ESS')
    write_json(output/'protocol.json', config)
    start = time.monotonic()
    write_json(output/'status.json', dict(status='initializing', started=stamp(), launcher_pid=os.getpid()))
    ctx = multiprocessing.get_context('spawn')
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=ctx) for _ in range(2)]
    monitor_log = (output/'gpu_metrics.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
                                '--format=csv', '-l', '10'], stdout=monitor_log, stderr=subprocess.DEVNULL)
    try:
        initialized = [pools[0].submit(initialize_lane, 0, [0,1], config),
                       pools[1].submit(initialize_lane, 1, [2], config)]
        audit = sorted([item for f in initialized for item in f.result()], key=lambda r:r['sector'])
        if len(set(item['parameter_sha256'] for item in audit)) != 1:
            raise RuntimeError('same-seed initial parameters differ across sectors')
        write_json(output/'initialization_audit.json', audit)
        print(json.dumps(dict(event='initialized', audit=audit)), flush=True)
        write_json(output/'status.json', dict(status='shared_burn', sweeps=config['burn']))
        collect(pools, sample_lane, config['burn'])
        from .mixture_sr import mixture_weights
        trace = []
        for step in range(1, config['pretrain_steps']+1):
            step_start = time.monotonic()
            samples = collect(pools, sample_lane, 2)
            x = np.concatenate([r['positions'] for r in samples])
            l = np.concatenate([r['layers'] for r in samples])
            evaluations = collect(pools, evaluate_lane, x, l)
            weights, diagnostics = mixture_weights(
                np.stack([r['log_density'] for r in evaluations]), [config['samples']//3]*3)
            rows = collect(pools, update_lane, weights, step, diagnostics)
            entry = dict(step=step, sectors=rows, mixture=diagnostics,
                         equal_weight_energy_per_particle_meV=float(np.mean([r['energy_per_particle_meV'] for r in rows])),
                         step_wall_seconds=time.monotonic()-step_start)
            trace.append(entry)
            write_json(output/'shared20_trace.json', trace)
            write_json(output/'status.json', dict(status='shared_training', step=step,
                                                  target=config['pretrain_steps'], latest=entry))
            print(json.dumps(entry), flush=True)
        fork_futures = [p.submit(fork_lane, x, l) for p in pools]
        forked = [f.result() for f in fork_futures]
        forks = [item for r in forked for item in r['forks']]
        if len(set(item['pool_sha256'] for item in forks)) != 1:
            raise RuntimeError('branch initial pools differ')
        if len(set(tuple(item['sampler_key']) for item in forks)) != 3:
            raise RuntimeError('branch sampler keys are not independent')
        write_json(output/'fork_audit.json', forked)
        for pool in pools:
            pool.shutdown(wait=True)
        pools = []
        write_json(output/'status.json', dict(status='private_training', target=config['branch_steps']))
        def lane0():
            return [run_branch(config, 0, 0), run_branch(config, 0, 1)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            p2 = executor.submit(run_branch, config, 1, 2)
            other = executor.submit(lane0)
            branches = [p2.result(), *other.result()]
        result = dict(status='complete' if all(r['status']=='complete' for r in branches) else 'finished_with_failures',
                      finished=stamp(), wall_seconds=time.monotonic()-start, branches=branches,
                      shared_gpu_lane_allocated_seconds=sum(r['gpu_lane_wall_seconds'] for r in forked),
                      branch_gpu_allocated_seconds=sum(r['wall_seconds'] for r in branches))
        write_json(output/'summary.json', result)
        write_json(output/'status.json', result)
    except BaseException as error:
        write_json(output/'status.json', dict(status='failed', finished=stamp(), error=repr(error)))
        raise
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_log.close()


if __name__ == '__main__':
    main()
