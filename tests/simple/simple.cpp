#include <weval.h>
#include <wizer.h>
#include <stdlib.h>
#include <stdio.h>

WIZER_DEFAULT_INIT();

enum Opcode {
    PushConst,
    Drop,
    Dup,
    GetLocal,
    SetLocal,
    Add,
    Sub,
    Print,
    Goto,
    GotoIf,
    UseOpViaMem,
    Exit,
};

struct Inst {
    Opcode opcode;
    uint32_t imm;

    explicit Inst(Opcode opcode_) : opcode(opcode_), imm(0) {}
    Inst(Opcode opcode_, uint32_t imm_) : opcode(opcode_), imm(imm_) {}
};

#define OPSTACK_SIZE 32
#define LOCAL_SIZE 32

struct State {
    uint32_t opstack[OPSTACK_SIZE];
    uint32_t locals[LOCAL_SIZE];
};

__attribute__((noinline)) void use_op_via_mem(uint32_t* operand) {
    *operand += 1;
}

bool Interpret(const Inst* insts, uint32_t ninsts, State* state) {
    insts = weval::assume_const_memory(insts);
    uint32_t pc = 0;
    uint32_t steps = 0;
    uint32_t* opstack = state->opstack;
    uint32_t* locals = state->locals;
    int sp = 0;
    opstack = weval::make_symbolic_ptr(opstack);
    locals = weval::make_symbolic_ptr(locals);

    // TODO: build an abstraction for `pc`: `InterpreterPC<u32>` (and
    // impls for u64 and T*) that have a `.loop([&](pc) { ... })`
    // method that returns either LoopResult::next_pc(pc) or
    // LoopResult::break_loop(val).
    weval::push_context(pc);
    while (true) {
        steps++;
        const Inst* inst = &insts[pc];
        pc++;
        weval::update_context(pc);
        switch (inst->opcode) {
            case PushConst:
                if (sp + 1 > OPSTACK_SIZE) {
                    return false;
                }
                opstack[sp++] = inst->imm;
                break;
            case Drop:
                if (sp == 0) {
                    return false;
                }
                sp--;
                break;
            case Dup:
                if (sp + 1 > OPSTACK_SIZE) {
                    return false;
                }
                if (sp == 0) {
                    return false;
                }
                opstack[sp] = opstack[sp - 1];
                sp++;
                break;
            case GetLocal:
                if (sp + 1 > OPSTACK_SIZE) {
                    return false;
                }
                if (inst->imm >= LOCAL_SIZE) {
                    return false;
                }
                opstack[sp++] = locals[inst->imm];
                break;
            case SetLocal:
                if (sp == 0) {
                    return false;
                }
                if (inst->imm >= LOCAL_SIZE) {
                    return false;
                }
                locals[inst->imm] = opstack[--sp];
                break;
            case Add:
                if (sp < 2) {
                    return false;
                }
                opstack[sp - 2] += opstack[sp - 1];
                sp--;
                break;
            case Sub:
                if (sp < 2) {
                    return false;
                }
                opstack[sp - 2] -= opstack[sp - 1];
                sp--;
                break;
            case Print:
                if (sp == 0) {
                    return false;
                }
                printf("%u\n", opstack[--sp]);
                break;
            case Goto:
                if (inst->imm >= ninsts) {
                    return false;
                }
                pc = inst->imm;
                weval::update_context(pc);
                break;
            case GotoIf:
                if (sp == 0) {
                    return false;
                }
                if (inst->imm >= ninsts) {
                    return false;
                }
                sp--;
                if (opstack[sp] != 0) {
                    pc = inst->imm;
                    weval::update_context(pc);
                    continue;
                }
                break;
            case UseOpViaMem:
                if (sp == 0) {
                    return false;
                }
                use_op_via_mem(&opstack[sp]);
                break;
            case Exit:
                goto out;
        }
    }
out:
    weval::pop_context();

    printf("Exiting after %d steps at PC %d.\n", steps, pc);
    return true;
}

Inst prog[] = {
    Inst(PushConst, 0),
    Inst(Dup),
    Inst(PushConst, 10000000),
    Inst(Sub),
    Inst(GotoIf, 6),
    Inst(Exit),
    Inst(PushConst, 1),
    Inst(Add),
    Inst(UseOpViaMem),
    Inst(Goto, 1),
};

typedef bool (*InterpretFunc)(const Inst* insts, uint32_t ninsts, State* state);

struct Func {
    const Inst* insts;
    uint32_t ninsts;
    InterpretFunc specialized;

    Func(const Inst* insts_, uint32_t ninsts_)
        : insts(insts_), ninsts(ninsts_), specialized(nullptr) {
        printf("ctor: ptr %p\n", &specialized);
        weval::weval(&specialized, Interpret, weval::Specialize(insts), weval::Specialize(ninsts), weval::Runtime<State*>());
    }

    bool invoke(State* state) {
        printf("Inspecting func ptr at: %p -> %p (size %lu)\n", &specialized, specialized, sizeof(specialized));
        if (specialized) {
            printf("Calling specialized function: %p\n", specialized);
            return specialized(insts, ninsts, state);
        }
        return Interpret(insts, ninsts, state);
    }
};

Func prog_func(prog, sizeof(prog)/sizeof(Inst));

int main(int argc, char** argv) {
    State* state = (State*)calloc(sizeof(State), 1);
    prog_func.invoke(state);
    fflush(stdout);
}
