from src.optimization.optimizer import (
    _classify, _loop_decision, _pick_winner, _commit_decision, CycleResult,
)

assert _classify(0.9, 0.75, 0.7, 0.6, 0.55) == 'crossed', _classify(0.9, 0.75, 0.7, 0.6, 0.55)
print('1 ok')
assert _classify(0.65, 0.60, 0.7, 0.6, 0.55) == 'improved', _classify(0.65, 0.60, 0.7, 0.6, 0.55)
print('2 ok')
assert _classify(0.64, 0.59, 0.7, 0.6, 0.55) == 'marginal', _classify(0.64, 0.59, 0.7, 0.6, 0.55)
print('3 ok')
assert _classify(0.6, 0.55, 0.7, 0.6, 0.55) == 'plateau', _classify(0.6, 0.55, 0.7, 0.6, 0.55)
print('4 ok')
assert _classify(0.5, 0.45, 0.7, 0.6, 0.55) == 'regressed', _classify(0.5, 0.45, 0.7, 0.6, 0.55)
print('5 ok')
print('classification OK')

assert _loop_decision('crossed', 1, 3, True)  == 'early_exit_crossed'
assert _loop_decision('crossed', 1, 3, False) == 'adopt'
assert _loop_decision('plateau', 1, 3, True)  == 'iterate_plateau'
assert _loop_decision('plateau', 2, 3, True)  == 'plateau_stop'
assert _loop_decision('regressed', 3, 3, True) == 'regress_max_cycles'
print('loop decisions OK')

c1 = CycleResult(cycle=1, classification='plateau', pass_rate=0.6, metric_avg=0.55,
                 delta_pass_pp=0, delta_metric=0, loop_decision='plateau_stop',
                 prompt_text='a', rationale='r', change_type='other', diff_lines=10,
                 validate_run_id='r1')
c2 = CycleResult(cycle=2, classification='improved', pass_rate=0.75, metric_avg=0.62,
                 delta_pass_pp=0.15, delta_metric=0.07, loop_decision='improved_max_cycles',
                 prompt_text='b', rationale='r2', change_type='other', diff_lines=20,
                 validate_run_id='r2')
w = _pick_winner([c1, c2])
assert w.cycle == 2, f'winner should be cycle 2, got {w.cycle}'
print('winner tie-break OK')

class _FakeRegistry:
    def __init__(self):
        self.registered = []
    def register(self, **kw):
        self.registered.append(kw)
        return 'v99'
    def update_metric_scores(self, *_, **__):
        return None

reg = _FakeRegistry()
status, ver, _ = _commit_decision(
    cycles=[c2], winner=c2, commit_on_plateau=False, role='coder', parent_version='v1',
    metric='step_efficiency', base_pass=0.6, base_metric=0.55, registry=reg, dry_run=False,
)
assert status == 'improved_below_threshold' and ver == 'v99', f'{status=} {ver=}'
print('aggressive (improved) OK')

reg2 = _FakeRegistry()
status, ver, _ = _commit_decision(
    cycles=[c1], winner=c1, commit_on_plateau=False, role='coder', parent_version='v1',
    metric='step_efficiency', base_pass=0.6, base_metric=0.55, registry=reg2, dry_run=False,
)
assert status == 'plateau_no_commit' and ver is None, f'{status=} {ver=}'
print('plateau no-commit OK')

reg3 = _FakeRegistry()
status, ver, _ = _commit_decision(
    cycles=[c1], winner=c1, commit_on_plateau=True, role='coder', parent_version='v1',
    metric='step_efficiency', base_pass=0.6, base_metric=0.55, registry=reg3, dry_run=False,
)
assert status == 'forced_plateau' and ver == 'v99', f'{status=} {ver=}'
print('plateau forced-commit OK')

c3 = CycleResult(cycle=1, classification='regressed', pass_rate=0.4, metric_avg=0.4,
                 delta_pass_pp=-0.2, delta_metric=-0.15, loop_decision='regress_max_cycles',
                 prompt_text='c', rationale='r3', change_type='other', diff_lines=30,
                 validate_run_id='r3')
reg4 = _FakeRegistry()
status, ver, _ = _commit_decision(
    cycles=[c3], winner=c3, commit_on_plateau=True, role='coder', parent_version='v1',
    metric='step_efficiency', base_pass=0.6, base_metric=0.55, registry=reg4, dry_run=False,
)
assert status == 'all_regressed' and ver is None, f'{status=} {ver=}'
assert reg4.registered == []
print('regressed-never-commit OK')

c4 = CycleResult(cycle=1, classification='crossed', pass_rate=0.9, metric_avg=0.85,
                 delta_pass_pp=0.3, delta_metric=0.3, loop_decision='early_exit_crossed',
                 prompt_text='d', rationale='r4', change_type='other', diff_lines=5,
                 validate_run_id='r4')
reg5 = _FakeRegistry()
status, ver, _ = _commit_decision(
    cycles=[c4], winner=c4, commit_on_plateau=False, role='coder', parent_version='v1',
    metric='step_efficiency', base_pass=0.6, base_metric=0.55, registry=reg5, dry_run=True,
)
assert status == 'dry_run' and ver is None, f'{status=} {ver=}'
assert reg5.registered == []
print('dry_run never-commit OK')

print('\nAll decision-tree tests passed.')
