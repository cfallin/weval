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
    int opstack_len;
    uint32_t locals[LOCAL_SIZE];
};

bool Interpret(const Inst* insts, uint32_t ninsts, State* state) {
    insts = weval::assume_const_memory(insts);
    uint32_t pc = 0;
    uint32_t steps = 0;
    uint32_t* opstack = state->opstack;
    uint32_t* locals = state->locals;
    int opstack_len = state->opstack_len;
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
                if (opstack_len + 1 > OPSTACK_SIZE) {
                    return false;
                }
                opstack[opstack_len++] = inst->imm;
                break;
            case Drop:
                if (opstack_len == 0) {
                    return false;
                }
                opstack_len--;
                break;
            case Dup:
                if (opstack_len + 1 > OPSTACK_SIZE) {
                    return false;
                }
                if (opstack_len == 0) {
                    return false;
                }
                opstack[opstack_len] = opstack[opstack_len - 1];
                opstack_len++;
                break;
            case GetLocal:
                if (opstack_len + 1 > OPSTACK_SIZE) {
                    return false;
                }
                if (inst->imm >= LOCAL_SIZE) {
                    return false;
                }
                opstack[opstack_len++] = locals[inst->imm];
                break;
            case SetLocal:
                if (opstack_len == 0) {
                    return false;
                }
                if (inst->imm >= LOCAL_SIZE) {
                    return false;
                }
                locals[inst->imm] = opstack[--opstack_len];
                break;
            case Add:
                if (opstack_len < 2) {
                    return false;
                }
                opstack[opstack_len - 2] += opstack[opstack_len - 1];
                opstack_len--;
                break;
            case Sub:
                if (opstack_len < 2) {
                    return false;
                }
                opstack[opstack_len - 2] -= opstack[opstack_len - 1];
                opstack_len--;
                break;
            case Print:
                if (opstack_len == 0) {
                    return false;
                }
                printf("%u\n", opstack[--opstack_len]);
                break;
            case Goto:
                if (inst->imm >= ninsts) {
                    return false;
                }
                pc = inst->imm;
                weval::update_context(pc);
                break;
            case GotoIf:
                if (opstack_len == 0) {
                    return false;
                }
                if (inst->imm >= ninsts) {
                    return false;
                }
                opstack_len--;
                if (opstack[opstack_len] != 0) {
                    pc = inst->imm;
                    weval::update_context(pc);
                    continue;
                }
                break;
            case Exit:
                goto out;
        }
    }
out:
    weval::pop_context();
    state->opstack_len = opstack_len;
    weval::flush_to_mem(opstack, OPSTACK_SIZE);
    weval::flush_to_mem(locals, LOCAL_SIZE);

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
    Inst(Goto, 1),
};

typedef bool (*InterpretFunc)(const Inst* insts, uint32_t ninsts, State* state);

struct Func {
    const Inst* insts;
    uint32_t ninsts;
    InterpretFunc specialized;

    Func(const Inst* insts_, uint32_t ninsts_)
        : insts(insts_), ninsts(ninsts_), specialized(nullptr) {
        weval::weval(&specialized, Interpret, weval::Specialize(insts), weval::Specialize(ninsts), weval::Runtime<State*>());
    }

    bool invoke(State* state) {
        printf("Inspecting func ptr at: %p\n", &specialized);
        if (specialized) {
            printf("Calling specialized function: %p\n", specialized);
            return specialized(insts, ninsts, state);
        }
        return Interpret(insts, ninsts, state);
    }
};

Func prog_func(prog, sizeof(prog)/sizeof(Inst));

int main() {
    State* state = (State*)calloc(sizeof(State), 1);
    prog_func.invoke(state);
    fflush(stdout);
}
