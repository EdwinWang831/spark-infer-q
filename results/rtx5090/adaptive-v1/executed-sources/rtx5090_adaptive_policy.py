"""Causal history-only controls and a small adaptation of pinned DFlash generation."""
import ast
import hashlib
import inspect
import time


class Controller:
    def __init__(self, config):
        self.config = config
        self.last = None
        self.ewma = None
        self.low_streak = 0
        self.after_skip = False
        self.actions = []
        self.wall_ns = 0

    def choose(self):
        begin = time.perf_counter_ns()
        kind = self.config['kind']
        if kind == 'fixed':
            width = self.config['width']
        elif kind == 'schedule':
            schedule = self.config['widths']
            width = schedule[len(self.actions) % len(schedule)]
        elif self.last is None or self.after_skip:
            width = 16
        elif kind == 'ewma':
            width = 8 if self.ewma <= 4 else 16
        elif kind == 'lagged':
            width = 8 if self.last <= 4 else 16
        elif kind == 'candidate':
            if self.low_streak >= self.config['skip_patience']:
                width = 1
            else:
                width = 8 if self.last <= self.config['short_threshold'] else 16
        else:
            raise ValueError(f'unknown controller: {kind}')
        self.actions.append(width)
        self.wall_ns += time.perf_counter_ns() - begin
        return width

    def observe(self, width, progress):
        begin = time.perf_counter_ns()
        self.after_skip = width == 1
        if width == 1:
            self.low_streak = 0
        else:
            self.last = progress
            self.ewma = progress if self.ewma is None else .5 * progress + .5 * self.ewma
            self.low_streak = self.low_streak + 1 if progress <= 1 else 0
        self.wall_ns += time.perf_counter_ns() - begin


def check_frontier(target_cache, draft_cache, hidden, start):
    if target_cache.get_seq_length() != start:
        raise RuntimeError('incorrect target cache frontier')
    if draft_cache.get_seq_length() + hidden.shape[1] != start:
        raise RuntimeError('draft cache plus pending features does not reach target frontier')


def compile_generate(upstream):
    original = inspect.unwrap(upstream)
    parsed = ast.parse(inspect.getsource(original))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef))
    function.name = '_adaptive_generate'
    function.decorator_list = []
    function.args.kwonlyargs += [ast.arg(arg='controller_config'), ast.arg(arg='audit')]
    function.args.kw_defaults += [None, ast.Constant(False)]
    loops = [node for node in function.body if isinstance(node, ast.While)]
    if len(loops) != 1:
        raise ValueError('unexpected upstream loop structure')
    loop = loops[0]
    target_calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'target']
    if len(target_calls) != 1:
        raise ValueError('expected one verification call')
    keyword = next(kw for kw in target_calls[0].keywords if kw.arg == 'output_hidden_states')
    if ast.unparse(keyword.value) != 'verify_size > 1':
        raise ValueError('unexpected upstream feature rule')
    keyword.value = ast.Constant(True)
    # Initialize features for every mode, including a first-round skip. The
    # external baseline remains the unmodified official generator.
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'block_size' for t in node.targets))
    function.body[index] = ast.parse('block_size = 16').body[0]
    function.body.insert(index + 1, ast.parse('_controller = Controller(controller_config)').body[0])
    loop.body = ast.parse('block_size = _controller.choose()').body + loop.body
    loop.body += ast.parse('''
if verify_size == 1:
    target_hidden = torch.cat((target_hidden, extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :produced, :]), dim=1)
_controller.observe(block_size, produced)
if audit:
    check_frontier(past_key_values_target, past_key_values_draft, target_hidden, start)
''').body
    returned = function.body[-1]
    if not isinstance(returned, ast.Return) or not isinstance(returned.value, ast.Call):
        raise ValueError('unexpected upstream stats return')
    returned.value.keywords += [
        ast.keyword(arg='action_trace', value=ast.parse('_controller.actions', mode='eval').body),
        ast.keyword(arg='controller_wall_ns', value=ast.parse('_controller.wall_ns', mode='eval').body),
    ]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(original.__globals__, Controller=Controller, check_frontier=check_frontier)
    exec(compile(module, '<pinned-dflash-adaptive>', 'exec'), namespace)
    return namespace['_adaptive_generate'], hashlib.sha256(ast.unparse(module).encode()).hexdigest()
