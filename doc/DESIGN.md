# Design notes for weval/wick

- weval: WebAssembly (partial) Evaluator -- the tool
- wick: WebAssembly Interpreter Compilation frameworK

## Partial evaluation of WebAssembly

The abstraction that we provide is that we can *specialize* a
particular function f(x, y, z, ...) on some values of arguments. We
rewrite the entry-point of f to dispatch on the argument values we
have specialized. In other words, the instructions to the tool consist
of instructions of the form "specialize function f for argument values
(_, 1, _), (2, _, _), ...".

The result of specialization is another function body. During partial
evaluation a value on the operand stack, a local, a global, or memory
can be either concrete or symbolic. The specialization process is
total (cannot fail): as a fallback, the entire original function body
can be emitted to evaluate at runtime.

Memory can be updated during the partial evaluation. However, only
memory reachable from values that are concrete during specialization
(partial evaluation) can be read or written. Any other memory access
is symbolic and results in runtime evaluation. Globals can also be
read and updated.

The basic approach to specialization is to interpret the Wasm
bytecodes over the expanded value space (concrete or symbolic) --
i.e., perform an abstract interpretation. A symbolic value can be an
arbitrary Wasm expression AST.

## Control Flow

Control flow can be evaluated either symbolically or concretely. WHen
it cannot be resolved concretely, the fallback of control flow in the
final specialized function body is always available. 

### If/Else

This is the simplest case. If the condition to the if/else construct
is concretely known, we take the appropriate side of the conditional
and evaluate it as if it were inline at this point (i.e., we
constant-fold the branch). Otherwise we emit an if/else in the
specialized function (but we continue to specialize each body).

### Blocks and Forward Edges

When a block is introduced in the original function, we introduce an
equivalent block in the specialized function. On a br\_if, if the
condition is statically known, we can replace it with an unconditional
br and omit the rest of the body; otherwise we emit a br\_if in the
specialized function body as well.

### Loops, Unrolling, and Interpreter-Loop PC Points

This is the trickiest bit. We specialize the loop body first as if we
were specializing a block body, and we collect all backedge
conditions. We also look for any concrete calls to the
"interpreter-loop PC point" intrinsic.

There are three kinds of loops:

- Those with fully concrete loop-backedge conditions. If we hit a
  concrete loop backedge during the eval of a loop body, we have
  peeled off one iteration and we emit the runtime remainder of that
  partial evaluation into the specialized function body as a block,
  and evaluate the body again with the updated state.
  
- Those with an interpreter-loop PC point intrinsic call. If we hit
  one of these, then regardless of whether we hit a symbolic or
  concrete loop backedge, we will peel off the iteration and emit a
  "PC-mapped" control flow construct in the output (which implies an
  arbitrary CFG that is stackified back to blocks and loops).
  
  If we hit a symbolic backedge condition in this case, we fork the
  state, and add both branches (or all branches, for a br_table) to
  the workqueue. Each state fork should have a different concrete PC
  value and so will result in another PC entry in the PC-mapped block.
  
  Q: how to merge state? Actually we probably don't want to support
  any concrete writes to memory from other globals, and any locals
  that were modified in any forked state branch become symbolic. We
  want stack space to be "local scratch", and so the shadow stack
  pointer (global 0 usually) is used concretely.
  
- Those with symbolic loop-backedge conditions. If we hit a symbolic
  condition, we emit the loop itself into the specialized function
  body.
  
### Memory Renaming

- interpreter's operand stack. On store, if address is concrete

### Inlining

### Speculation
