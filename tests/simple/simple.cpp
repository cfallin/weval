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
    uint32_t locals[LOCAL_SIZE];
};

struct Func {
    const Inst* insts;
    uint32_t ninsts;
    void* specialized;

    Func(const Inst* insts_, uint32_t ninsts_);
    bool invoke(State* state);
};

bool Interpret(const Func* func_, State* state);

Func::Func(const Inst* insts_, uint32_t ninsts_)
    : insts(insts_), ninsts(ninsts_), specialized(nullptr) {
    weval::enroll(Interpret, this, 0, &specialized);
}

bool Func::invoke(State* state) {
    return Interpret(this, state);
}

bool Interpret(const Func* func_, State* state) {
    uint32_t pc = 0;
    uint32_t steps = 0;
    uint32_t* opstack = state->opstack;
    uint32_t* locals = state->locals;
    int sp = 0;
    opstack = weval::make_symbolic_ptr(opstack);
    locals = weval::make_symbolic_ptr(locals);

    const Func* func = weval::start(func_, pc, &func->specialized);
    const Inst* insts = weval::assume_const_memory(func->insts);
    while (true) {
        steps++;
        const Inst* inst = &insts[pc];
        pc++;
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
                if (inst->imm >= func->ninsts) {
                    return false;
                }
                pc = inst->imm;
                break;
            case GotoIf:
                if (sp == 0) {
                    return false;
                }
                if (inst->imm >= func->ninsts) {
                    return false;
                }
                sp--;
                if (opstack[sp] != 0) {
                    pc = inst->imm;
                }
                break;
            case Exit:
                goto out;
        }

        pc = weval::pc_ctx(pc);
    }
out:
    weval::end();
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

Func prog_func(prog, sizeof(prog)/sizeof(Inst));

int main(int argc, char** argv) {
    State* state = (State*)calloc(sizeof(State), 1);
    prog_func.invoke(state);
    fflush(stdout);
}
