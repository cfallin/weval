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

    pc = weval::loop_pc(pc);
    while (true) {
        const Inst* inst = &insts[pc];
        pc++;
        weval::loop_end();
        weval::loop_pc(pc);
        switch (inst->opcode) {
            case PushConst:
                if (state->opstack_len + 1 > OPSTACK_SIZE) {
                    return false;
                }
                state->opstack[state->opstack_len++] = inst->imm;
                break;
            case Drop:
                if (state->opstack_len == 0) {
                    return false;
                }
                state->opstack_len--;
                break;
            case Dup:
                if (state->opstack_len + 1 > OPSTACK_SIZE) {
                    return false;
                }
                if (state->opstack_len == 0) {
                    return false;
                }
                state->opstack[state->opstack_len] = state->opstack[state->opstack_len - 1];
                state->opstack_len++;
                break;
            case GetLocal:
                if (state->opstack_len + 1 > OPSTACK_SIZE) {
                    return false;
                }
                if (inst->imm >= LOCAL_SIZE) {
                    return false;
                }
                state->opstack[state->opstack_len++] = state->locals[inst->imm];
                break;
            case SetLocal:
                if (state->opstack_len == 0) {
                    return false;
                }
                if (inst->imm >= LOCAL_SIZE) {
                    return false;
                }
                state->locals[inst->imm] = state->opstack[--state->opstack_len];
                break;
            case Add:
                if (state->opstack_len < 2) {
                    return false;
                }
                state->opstack[state->opstack_len - 2] += state->opstack[state->opstack_len - 1];
                state->opstack_len--;
                break;
            case Sub:
                if (state->opstack_len < 2) {
                    return false;
                }
                state->opstack[state->opstack_len - 2] -= state->opstack[state->opstack_len - 1];
                state->opstack_len--;
                break;
            case Print:
                if (state->opstack_len == 0) {
                    return false;
                }
                printf("%u\n", state->opstack[--state->opstack_len]);
                break;
            case Goto:
                if (inst->imm >= ninsts) {
                    return false;
                }
                pc = inst->imm;
                weval::loop_end();
                weval::loop_pc(pc);
                break;
            case GotoIf:
                if (state->opstack_len == 0) {
                    return false;
                }
                if (inst->imm >= ninsts) {
                    return false;
                }
                state->opstack_len--;
                if (state->opstack[state->opstack_len] != 0) {
                    pc = inst->imm;
                    weval::loop_end();
                    weval::loop_pc(pc);
                }
                break;
            case Exit:
                return true;
        }
    }
}

Inst prog[] = {
    Inst(PushConst, 0),
    Inst(Dup),
    Inst(PushConst, 10),
    Inst(Sub),
    Inst(GotoIf, 6),
    Inst(Exit),
    Inst(Dup),
    Inst(Print),
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
        if (specialized) {
            return specialized(insts, ninsts, state);
        }
        return Interpret(insts, ninsts, state);
    }
};

Func prog_func(prog, sizeof(prog)/sizeof(Inst));

int main() {
    State* state = (State*)calloc(sizeof(State), 1);
    prog_func.invoke(state);
}
