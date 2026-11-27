
# Algorithm API v2 — Contract and Development Roadmap

**Project:** WDS Optimizer  
**Status:** Living technical specification  
**Target:** v1.0.0  
**Last updated:** 2026-11-28

---

## 1. Purpose and Product Vision

### 1.1 Primary Product Goal

The primary goal of this development effort is to provide a secure, transparent, and user-friendly plugin system that allows WDS Optimizer users to create, import, inspect, edit, validate, and execute their own optimization algorithms without modifying the application's core code.

Users should be able to work with optimization code directly through the GUI.

The intended user experience includes:

- Importing a custom algorithm from a file.
- Giving the algorithm a name and a meaningful description.
- Viewing and editing algorithm source code through an integrated code editor.
- Managing built-in and custom algorithms through clear, dedicated GUI dialogs.
- Validating custom algorithms before execution.
- Executing custom code in an isolated environment with appropriate security restrictions.
- Receiving meaningful validation errors, execution errors, and progress information.
- Reusing built-in algorithms as reference templates when developing custom algorithms.

The code editor should provide a professional editing experience, preferably similar to Monaco Editor, subject to Streamlit integration and implementation feasibility.

### 1.2 Constraint Handling Is Also Part of the Product Goal

The system must not make optimization algorithms the only user-visible extension point.

Constraint handling is another important part of the optimization process. Users should be able to understand how constraints and penalties influence the optimization process.

The GUI should therefore provide a clear and separate workflow for:

- Selecting and configuring constraint handling.
- Selecting and managing optimization algorithms.
- Inspecting the relevant source code and descriptions.
- Understanding how algorithms interact with constraint handling.

Custom constraint policies are not part of the v1.0.0 public plugin scope. The initial implementation uses the supported built-in constraint policies.

### 1.3 Engineering Goal

Algorithm API v2 is the technical foundation of this plugin system.

It defines a stable boundary between:

1. User-provided algorithm code.
2. The optimization engine.
3. Hydraulic simulation and evaluation.
4. GUI and CLI execution workflows.

The API must be minimal, explicit, testable, and independent of GUI implementation details.

### 1.4 Architectural Principles

- User algorithms must not depend on internal implementation details.
- Algorithms must not directly access WNTR, EPANET, hydraulic solvers, or internal caches.
- All candidate evaluation must go through the public Evaluator API.
- Randomness, progress reporting, and cancellation must be provided through AlgorithmContext.
- Built-in and custom algorithms must use the same public API.
- Plugin execution must be isolated from the main application process.
- Security must not rely on `exec`, restricted namespaces, or process separation alone.
- The initial release should prioritize a reliable and understandable plugin workflow over unnecessary extensibility.

---

## 2. Status Definitions

Each topic uses one of the following statuses:

- **PROPOSED:** Suggested but not yet approved.
- **ACCEPTED:** Explicitly approved as part of the contract.
- **IMPLEMENTED:** Implemented and verified through relevant tests.
- **OPEN:** Requires further investigation or a decision.

A topic may have both an architectural status and an implementation status.

For example, a design can be ACCEPTED while its implementation remains incomplete.

---

## 3. Public Algorithm API

### 3.1 Algorithm Entrypoint

**Status: ACCEPTED**

Every algorithm must expose the following entrypoint:

```python
def run(problem, config, *, evaluator, context):
    ...
    return AlgorithmResult(...)
```

Arguments:

- `problem`: Public, read-only description of the optimization problem.
- `config`: Immutable algorithm configuration.
- `evaluator`: The only public interface for evaluating and comparing candidates.
- `context`: Random number generator, progress reporting, and cancellation interface.

Return value:

- `AlgorithmResult`

The algorithm must not access internal Core objects, WNTR, EPANET, hydraulic solvers, or internal caches.

### 3.2 Public API Boundary

**Status: ACCEPTED**

The public algorithm API consists of:

- OptimizationProblem
- PipeInfo
- AlgorithmConfig
- EvaluationResult
- ConvergenceHistory
- AlgorithmResult
- Evaluator
- AlgorithmContext
- OptimizationCancelled
- Algorithm Metadata

Internal implementation classes must not become public plugin dependencies without an explicit contract decision.

---

## 4. Data Models

### 4.1 OptimizationProblem

**Status: ACCEPTED**

```python
@dataclass(frozen=True)
class OptimizationProblem:
    pipes: tuple[PipeInfo, ...]
    diameters: np.ndarray
    junction_names: tuple[str, ...]
    units: str
```

### 4.2 PipeInfo

**Status: ACCEPTED**

```python
@dataclass(frozen=True)
class PipeInfo:
    name: str
    is_variable: bool
    initial_diameter: float | None = None
```

Rules:

- `pipes` preserve the original network order.
- Variable pipes are identified through `is_variable`.
- `variable_pipe_names` defines candidate gene order.
- `initial_diameter` is descriptive information, not an initial candidate.
- No `initial_candidate` is exposed.

### 4.3 Candidate Representation

**Status: ACCEPTED**

A candidate is a one-dimensional integer vector.

Each element represents a zero-based index into the available diameter list.

Candidate length:

`NVar = number of variable pipes`

Each gene must satisfy:

`0 <= candidate[i] < ND`

where `ND` is the number of available diameters.

The candidate contains genes only for variable pipes.

### 4.4 Problem Data Isolation

**Status: ACCEPTED**

The public problem object must not expose:

- `wn`
- Hydraulic solver instances
- Internal caches
- Internal optimization arrays or dictionaries
- `InitialD`
- `FixedPipes`
- `VariablePipes`

Public arrays must be independent copies and read-only.

### 4.5 EvaluationResult

**Status: ACCEPTED**

```python
@dataclass(frozen=True)
class EvaluationResult:
    cost: float
    violation: float
    feasible: bool
    converged: bool
    pressures: np.ndarray | None = None
    velocities: np.ndarray | None = None
```

Rules:

- `feasible=True` implies `converged=True` and `violation=0`.
- A converged result must have a finite cost.
- A non-converged result uses `cost=inf`.
- NaN and negative infinity are invalid.
- Hydraulic arrays are independent read-only copies.
- Pressure order follows `problem.junction_names`.
- Velocity order follows `problem.pipe_names`.
- Units follow `problem.units`.

Failure sentinel:

- `cost = inf`
- `violation = INVALID_VIOLATION`
- `feasible = False`
- `converged = False`
- `pressures = None`
- `velocities = None`

`INVALID_VIOLATION = 1e8`.

Hydraulic arrays are optional in the public result. Their default behavior is controlled by `return_hydraulics=False`.

Constraint calculations must still be performed when hydraulic arrays are not requested by the algorithm.

### 4.6 ConvergenceHistory

**Status: ACCEPTED**

```python
@dataclass
class ConvergenceHistory:
    best_cost: np.ndarray
    best_violation: np.ndarray
    best_feasible: np.ndarray
```

Rules:

- The history is mutable during algorithm execution.
- All arrays have the same length.
- Each row describes the same selected incumbent.
- `best_feasible` describes the feasibility of the selected incumbent, not whether any feasible candidate has ever been found.
- `snapshot()` returns an independent copy.

### 4.7 AlgorithmResult

**Status: ACCEPTED**

The result contains:

- Best candidate and its evaluation.
- Convergence history snapshot.
- Algorithm-specific diagnostics.

Rules:

- `best_candidate` and `best_evaluation` must either both be present or both be `None`.
- Candidate length and gene ranges are validated by Core.
- Generic restart diagnostics are not part of the public contract.

Core constructs the final diagnostics object:

```python
{
    "nfe": evaluator.nfe,
    "n_sim": evaluator.n_sim,
    "runtime_seconds": 42.6,
    "algorithm_specific": {}
}
```

Definitions:

- `nfe`: Number of valid candidate evaluation requests, including cache hits.
- `n_sim`: Number of actual hydraulic solver executions for candidate evaluations.
- `runtime_seconds`: Overall algorithm execution time, measured by the runner.
- `algorithm_specific`: Optional diagnostics provided by the algorithm.

The evaluator owns evaluation counters. The runner owns runtime measurement. Core finalizes the diagnostics.

### 4.8 AlgorithmConfig

**Status: IMPLEMENTED**

```python
@dataclass(frozen=True)
class AlgorithmConfig:
    seed: int
    population_size: int
    max_generations: int
    params: Mapping[str, Any]
```

Rules:

- `seed` must be a non-negative integer.
- `population_size` and `max_generations` must be positive integers.
- Boolean values are not accepted where integers are required.
- `params` contains algorithm-specific parameters only.
- Reserved configuration names are not allowed in `params`.
- The configuration is immutable.
- The parameter mapping is an independent read-only copy.
- NumPy scalar values are normalized.
- Supported parameter values are scalar values: string, boolean, integer, and finite float.
- Parameter-specific validation belongs to Metadata and the configuration builder/validator, not the generic config model.

Implementation detail:

A tuple-backed `_FrozenMapping` is used instead of `MappingProxyType` because configuration objects must support pickling across process boundaries.

Pickle round-trip must preserve immutability.

---

## 5. Algorithm Context

**Status: IMPLEMENTED**

Public interface:

```python
class OptimizationCancelled(Exception):
    def __init__(self, partial_result=None):
        ...

class AlgorithmContext:
    def __init__(
        self,
        *,
        rng,
        progress_callback=None,
        cancellation_check=None,
    ):
        ...

    @property
    def rng(self):
        ...

    def report_progress(
        self,
        *,
        generation,
        max_generations,
        metrics=None,
    ):
        ...

    def check_cancelled(self):
        ...
```

Rules:

- RNG must be an instance of `numpy.random.Generator`.
- The Runner supplies the RNG.
- Algorithms must use `context.rng` for randomness.
- Progress reporting must not depend on GUI frameworks.
- Progress generation must be an integer in the valid range.
- Metrics must be a mapping or `None`.
- `check_cancelled()` raises `OptimizationCancelled` when cancellation is requested.
- The context must not expose internal GUI or solver objects.

Partial-result handling remains a separate design topic.

---

## 6. Algorithm Metadata

**Status: IMPLEMENTED**

Required fields:

- `api_version`
- `short_id`
- `display_name`
- `category`
- `paper_faithful`
- `description`
- `references`
- `params`
- `constraint_policy`

Optional fields:

- `author`
- `version`
- `tags`

### 6.1 Algorithm Identifier

`short_id`:

- Uses uppercase letters, digits, and underscores.
- Must not start with a digit.
- Must be unique across registered plugins.
- Uniqueness is enforced by the Registry, not by the individual metadata validator.

### 6.2 Parameter Metadata

Required parameter fields:

- `key`
- `name`
- `display_name`
- `type`
- `default`
- `description`

Optional fields:

- `min`
- `max`
- `choices`
- `maps_to`

Supported types:

- `int`
- `float`
- `bool`
- `str`
- `choice`

Rules:

- Choice parameters require a non-empty choices definition.
- Choice defaults must belong to the allowed choices.
- Numeric bounds must be compatible with the parameter type.
- Boolean values must not be treated as numeric values.
- Unknown fields are rejected.
- Duplicate parameter keys are rejected.
- Metadata validation returns an independent normalized copy.

### 6.3 Constraint Policies

Supported built-in policy identifiers:

- `deb_prefers`
- `fixed_penalty_prefers`
- `simple_cost_only_prefers`

Custom constraint policies are outside the v1.0.0 public plugin scope.

---

## 7. Evaluator API

**Status: ACCEPTED — implementation pending verification**

### 7.1 Public Methods

```python
result = evaluator.evaluate(
    candidate,
    return_hydraulics=False,
)

results = evaluator.evaluate_population(population)

better = evaluator.prefers(result_a, result_b)
```

Read-only properties:

```python
evaluator.nfe
evaluator.n_sim
```

### 7.2 Single Candidate Evaluation

`evaluate()` accepts a one-dimensional integer candidate.

Validation rules:

- Candidate must be one-dimensional.
- Candidate length must equal `NVar`.
- Each gene must be an integer index in `[0, ND)`.
- Boolean and floating-point candidate arrays are rejected.
- Invalid inputs must raise `TypeError` or `ValueError`.
- Invalid input must be rejected before counters are incremented or solver work begins.

### 7.3 Population Evaluation

`evaluate_population()` accepts a two-dimensional population:

`shape = [NVar, NPopulation]`

Rules:

- Each column represents one candidate.
- `NPopulation >= 1`.
- The entire population must be validated before any candidate evaluation starts.
- One invalid column rejects the entire request.
- No partial solver work is allowed for invalid population input.
- Results are returned as a tuple in input order.

### 7.4 Cancellation

- Single evaluation checks cancellation before evaluation.
- Population evaluation checks cancellation before starting and between candidates.
- Cancellation raises `OptimizationCancelled`.
- A cancelled population evaluation never returns partial results.
- Counters retain the actual work completed before cancellation.

### 7.5 Candidate Comparison

`prefers(result_a, result_b)`:

- Returns a pure boolean.
- Uses the policy fixed when the evaluator is constructed.
- Does not change policy during a run.
- Returns `False` for ties.

Policy definitions:

**deb_prefers**

1. Feasible is preferred over infeasible.
2. Between infeasible candidates, lower violation is preferred.
3. Between feasible candidates, lower cost is preferred.

**fixed_penalty_prefers**

Uses a fixed penalty factor of `1e9`.

**simple_cost_only_prefers**

Compares cost only and may prefer an infeasible candidate.

### 7.6 Evaluation Counters

`nfe`:

- Counts every valid candidate evaluation request.
- Includes cache hits.
- Does not include invalid inputs.

`n_sim`:

- Counts actual solver executions for candidate evaluations.
- Includes failed real solver executions.
- Does not count cache hits.
- Counts both executions if a fast solver and a WNTR fallback both execute for the same candidate.
- Excludes startup validation simulations.

Both counters are read-only to algorithms.

Only Evaluator may update them.

### 7.7 Exception Policy

- Invalid public inputs are rejected with appropriate validation exceptions.
- Known hydraulic nonconvergence is converted into the public failure sentinel.
- Unexpected programming or infrastructure exceptions must propagate.
- The exact classification of known hydraulic errors must be based on verified WNTR/EPANET behavior, not broad exception catching or string matching.

---

## 8. Hydraulic Execution and Failure Classification

**Status: ACCEPTED in principle — exact EPANET classification OPEN**

### 8.1 Internal Hydraulic Result

A private internal representation is proposed:

```python
@dataclass(frozen=True)
class HydraulicResult:
    pressures: np.ndarray | None
    velocities: np.ndarray | None
    converged: bool
```

This object is not exposed to plugins.

It may be extended with a private failure code if actual implementation requires it.

### 8.2 Three Hydraulic Outcomes

**Outcome A — Successful hydraulic execution**

- Solver completes successfully.
- Pressure and velocity arrays are finite.
- `converged=True`.
- Raw hydraulic results may be cached.

**Outcome B — Known hydraulic failure**

- A verified hydraulic failure or nonconvergence occurs.
- Public result uses the failure sentinel.
- Failure may be cached because it is a valid outcome for that candidate.

**Outcome C — Unexpected software or infrastructure exception**

- Programming errors, invalid internal state, unexpected I/O failures, and unrelated infrastructure errors must propagate.
- Such failures must not be converted into normal hydraulic nonconvergence.
- They must not be cached as hydraulic results.

### 8.3 Pressure and Numerical Validity

- Non-finite pressure or velocity arrays represent hydraulic failure.
- Negative pressure alone is not a hydraulic solver failure.
- Finite negative pressures must be passed to constraint evaluation.
- Constraint handling determines their violation and feasibility.

### 8.4 EPANET Error Classification

The implementation must distinguish actual hydraulic failures from warnings and unrelated EPANET errors.

The exact classification of EPANET error codes and WNTR exception behavior must be verified against the project's installed WNTR/EPANET versions.

Do not classify every `EpanetException` as a hydraulic failure.

Do not use exception message string matching as the primary classification mechanism.

### 8.5 Existing Implementation Issues to Resolve

The legacy hydraulic implementation requires review:

- `simulation.py` currently catches broad `Exception` and returns failure.
- `fast_solver.py` currently catches `EpanetException` broadly.
- Error-code information may be cleared before classification.
- Fast solver initialization and validation currently use broad exception handling and may silently fall back to WNTR.

These behaviors must be revised to preserve the distinction between hydraulic failures and unexpected errors.

---

## 9. Hydraulic Cache

**Status: ACCEPTED — implementation pending verification**

### 9.1 Cache Scope

Each independent optimization Run must own a fresh cache.

No shared cache across:

- Separate Run operations.
- Sweep combinations.
- Sensitivity Analysis combinations.

No persistent disk cache is planned for v1.0.0.

The cache exists to avoid repeated hydraulic simulations within one optimization run, not to make separate runs artificially faster.

### 9.2 Cache Contents

The cache stores raw hydraulic results, not final constraint violation or feasibility.

It may store:

- Raw pressure arrays.
- Raw velocity arrays.
- Hydraulic convergence status.
- Other hydraulic-only data if implementation requires it.

Cost is computed from design and cost data independently of hydraulic simulation.

Violation and feasibility must be recomputed using the current constraint configuration.

### 9.3 Cache Key

The cache key is the candidate vector only:

```python
cache_key = tuple(int(x) for x in candidate)
```

`return_hydraulics` is not part of the key.

The cache stores raw hydraulic output, so the same candidate can reuse the hydraulic result regardless of whether the caller requests hydraulic arrays in its public result.

### 9.4 Cache Counters

- Every valid evaluation request increments `nfe`, including cache hits.
- Cache hits do not increment `n_sim`.
- Actual solver executions increment `n_sim`.
- Startup validation simulations are excluded.

### 9.5 Capacity and Eviction

No configurable cache capacity or additional eviction complexity is planned for v1.0.0.

The cache is conceptually unbounded within a single Run.

Memory behavior should be measured before introducing capacity limits or eviction policies.

---

## 10. Plugin Contract and Isolation

**Status: ACCEPTED as a product requirement — implementation design OPEN**

### 10.1 General Requirements

Custom algorithms must use the same public API as built-in algorithms.

The plugin must not access:

- WNTR objects.
- EPANET objects.
- Internal hydraulic solvers.
- Core internals.
- Internal caches.
- GUI session state.
- Private optimization data structures.

The only supported interaction with candidate evaluation is through the Evaluator.

### 10.2 Plugin Lifecycle

The intended lifecycle is:

1. Import plugin source.
2. Parse and validate metadata.
3. Validate the public entrypoint.
4. Validate configuration and parameter definitions.
5. Execute validation checks in an isolated environment.
6. Make the plugin available for selection.
7. Execute the algorithm through the standard Runner.
8. Collect results, diagnostics, progress, and errors.
9. Return results to the application.

Exact lifecycle details remain subject to implementation design.

### 10.3 Security Requirements

Process isolation alone is not considered a complete security boundary.

A restricted namespace or restricted `exec` environment is not sufficient.

The sandbox design must evaluate and enforce appropriate restrictions for:

- CPU time.
- Wall-clock execution time.
- Memory consumption.
- Filesystem access.
- Network access.
- Process creation.
- Access to environment variables and secrets.
- Access to the main application process.
- Inter-process communication.
- Plugin import and execution.

The sandbox must fail safely when resource limits are exceeded or execution is cancelled.

The exact sandbox technology and operating-system strategy remain OPEN.

### 10.4 Built-in Algorithms as Reference Templates

**Status: ACCEPTED**

Built-in algorithms should eventually be presented as reference templates rather than requiring a separate plugin execution architecture.

The intended template documentation includes:

- Overview.
- Algorithm logic.
- API usage.
- Parameters.
- Source code.
- References.

Users should be able to inspect the implementation and use it as a starting point for custom algorithms.

Built-in algorithms and custom algorithms must ultimately share the same public API, evaluator, context, and runner.

---

## 11. GUI Plugin Experience

**Status: ACCEPTED at conceptual level — detailed UI implementation OPEN**

### 11.1 Algorithm Tab Organization

The current `3. Algorithm` tab is too crowded because both Constraint Handling and Algorithms to Run contain plugin-related functionality.

The target is a clearer layout with separate entry points:

- Constraint Handling
- Algorithms to Run

Each should open a dedicated management dialog.

### 11.2 Algorithm Management Dialog

The intended dialog should support:

- Selecting built-in or custom algorithms.
- Importing a user algorithm.
- Naming the algorithm.
- Writing a detailed description.
- Viewing and editing source code.
- Validating the plugin.
- Saving a custom algorithm.
- Managing previously imported algorithms.

### 11.3 Constraint Handling Dialog

Constraint handling should have a similarly clear and consistent workflow.

The v1.0.0 scope uses built-in policies. Custom constraint policies are not included in the initial public plugin contract.

### 11.4 Code Editor

The GUI should provide a proper code editor.

A Monaco-like editing experience is preferred, but the final implementation must consider:

- Streamlit compatibility.
- Python syntax highlighting.
- Editing and validation workflow.
- Source-code loading and saving.
- Error display.
- Safe interaction with the plugin execution environment.

The specific editor component remains OPEN until integration is evaluated.

---

## 12. Built-in Algorithms

**Status: ACCEPTED as a direction — migration pending**

The current built-in algorithms are:

- GA
- PSO
- DE
- SADE

SADE is the paper-faithful implementation of Zheng, Zecchin & Simpson (2013).

The current GA, PSO, and DE implementations are not claimed to be paper-faithful.

The goal is to migrate built-in algorithms to the same API v2 contract used by custom plugins.

Public restart-on-stagnation functionality has been removed from the intended v1.0.0 design.

Custom algorithms may implement their own restart mechanisms internally if desired.

---

## 13. Implementation Status

This section records implementation evidence separately from architectural decisions.

### 13.1 Implemented and Tested

| Component | Status | Evidence |
|---|---|---|
| Public data models | IMPLEMENTED | `tests/test_algorithm_api_models.py` |
| OptimizationProblem and PipeInfo | IMPLEMENTED | Model tests |
| EvaluationResult | IMPLEMENTED | Model tests |
| ConvergenceHistory | IMPLEMENTED | Model tests |
| AlgorithmResult | IMPLEMENTED | Model tests |
| AlgorithmConfig | IMPLEMENTED | `tests/test_algorithm_api_config.py` |
| AlgorithmContext | IMPLEMENTED | `tests/test_algorithm_api_context.py` |
| Algorithm Metadata | IMPLEMENTED | `tests/test_algorithm_api_metadata.py` |

Last reported test results:

- `tests/test_algorithm_api_config.py`: 49 passed.
- API model, context, metadata, and config test group: 224 passed.
- Full test suite: 531 passed, 8 deselected.

These results describe the last reported local test run and must be refreshed after subsequent changes.

### 13.2 Accepted but Implementation Not Yet Verified

- Evaluator API v2.
- Per-run hydraulic cache.
- Hydraulic result classification.
- Core integration.
- Standard Runner integration.
- Built-in algorithm migration.
- Plugin Registry and lifecycle.
- Isolated plugin execution.
- GUI plugin management dialogs.
- Integrated code editor.
- Reference template experience.

### 13.3 Known Legacy Areas Requiring Migration

- `simulation.py`
- `fast_solver.py`
- Legacy experiment runner cache behavior.
- Legacy sensitivity runner cache behavior.
- Existing algorithm execution interfaces.
- Existing GUI algorithm selection and plugin workflow.

The migration must preserve existing application functionality while ensuring that independent Runs do not share hydraulic caches.

---

## 14. Open Decisions and Investigations

The following topics must not be treated as finalized until explicitly resolved.

| Topic | Status | Next action |
|---|---|---|
| Exact WNTR/EPANET hydraulic error classification | OPEN | Inspect installed-version behavior and error codes |
| Fast solver fallback exception policy | OPEN | Distinguish compatibility failures from unexpected failures |
| Plugin sandbox technology | OPEN | Evaluate realistic process and resource isolation |
| Sandbox filesystem and network restrictions | OPEN | Define security boundary |
| Plugin timeout and memory limits | OPEN | Define enforceable limits |
| Plugin process communication | OPEN | Design serialization and IPC |
| Partial result behavior on cancellation | OPEN | Define ownership and lifecycle |
| Code editor integration | OPEN | Evaluate Streamlit-compatible editor options |
| Plugin storage and persistence | OPEN | Define source and metadata storage |
| Plugin validation workflow | OPEN | Define static and runtime validation |
| Built-in algorithm migration sequence | OPEN | Plan after API and Runner integration |

---

## 15. Development Workflow and Contract Governance

This document is the primary reference for Algorithm API v2 and the plugin-system architecture.

### 15.1 Working Protocol

At the beginning of each technical development session:

1. Read this document from the `main` branch of the GitHub repository.
2. Review accepted decisions and current implementation status.
3. Continue from the latest documented state.
4. Do not reopen accepted decisions without a concrete technical reason.

For every new design topic:

1. Explain the problem.
2. Present the proposed solution and alternatives when useful.
3. Discuss trade-offs.
4. Obtain explicit approval before treating a proposal as accepted.
5. After approval, specify the exact document changes.
6. The developer updates the document locally and commits/pushes it.
7. In the next session, verify the updated document from GitHub.

Uncommitted local changes are not considered part of the shared contract.

### 15.2 Implementation Workflow

Prefer small, independently testable implementation steps.

For each implementation step:

1. Identify the exact files to change.
2. Explain the intended behavior.
3. Implement the smallest coherent change.
4. Add or update tests.
5. Run focused tests.
6. Run broader tests when appropriate.
7. Review `git diff --check`.
8. Commit the verified change separately.

### 15.3 Decision Integrity

- ACCEPTED decisions must not silently change.
- Implementation details may evolve if they preserve the approved contract.
- Any change to public behavior requires explicit review.
- An implementation is not considered complete merely because its design was approved.
- Test evidence must be recorded separately from design approval.
- Security requirements must not be weakened for implementation convenience.

---

## 16. Release Scope and Priorities

The primary objective is a reliable v1.0.0 release.

Priorities:

1. Stable public Algorithm API v2.
2. Correct and predictable hydraulic evaluation.
3. Per-run cache isolation.
4. Reliable cancellation and progress reporting.
5. Secure and controlled custom algorithm execution.
6. Clear GUI workflow for algorithm and constraint handling.
7. Practical source-code editing and plugin management.
8. Built-in algorithms migrated to the same public contract.

Avoid unnecessary complexity, premature extensibility, and features that do not materially improve the v1.0.0 product.

The system should remain extensible, but the initial release must be understandable, testable, and dependable.
