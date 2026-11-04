import random
from enum import Enum
PEs = 3
PE_LM_Size = 1024

# модель в формате torch:
# Sequential(nn.Sigmoid(), nn.Linear(9, 3), nn.Linear(3, 3))
model = [
    ('in',[[(1)<<4 for j in range(9)]]), # это вход
    ('sig',[[0xef for i in range(9)]]), # sigmoid
    ('mul',[[1<<4 for j in range(9)] for i in range(3)]),
    ('mul', [[(-1<<4)&0xff,0,0,0][::-1],[0,(-1<<4)&0xff,0,0][::-1],[0,0,(-1<<4)&0xff,0][::-1]]) # инвертируем знак, унможив на -1
]

sm_layout = [[0 for j in range(4)] for i in range(1000)]

# Располагаем веса в shared memory
layers_indx_sm = []
sm_ind = 1
for _,layer in model:
    inds_layer = []
    for column in layer:
        kk = sm_ind
        for j in range(0,len(column), 4):
            if j+4 > len(column): # размер кусочка не кратен размеру слова
                sm_layout[sm_ind] = column[j:j+4] + [0 for i in range(4-(len(column)-j))]
            else:
                sm_layout[sm_ind] = column[j:j+4]
            sm_ind += 1
        inds_layer += [list(range(kk,sm_ind))]
    layers_indx_sm += [inds_layer]
input_addr = sm_ind

# print(layers_indx_sm)
# print(sm_layout[:20])

lm_ind = sm_ind + 1

free_memories = [[True for i in range(PE_LM_Size)] for pe in range(PEs)]
def get_first_free_addr_lm(pe_num):
    for i in range(PE_LM_Size):
        if free_memories[pe_num][i]:
            return i
    return -1
def alloc_lm(pe_num):
    addr = get_first_free_addr_lm(pe_num)
    assert addr != -1, "Попытка занять используемую память"
    free_memories[pe_num][addr] = False
    return addr
def free_lm(pe_num, addr):
    assert free_memories[pe_num][addr] == False, "Попытка освободить свободную память"
    free_memories[pe_num][addr] = True

class PE_State(Enum):
    FREE = 0
    MEM = 1
    OCCUPIED = 2

class PE_MUL_State(Enum):
    FREE = 0
    LOAD_OP1 = 1
    LOAD_OP2 = 2
    MUL = 3
    SUM = 4
    ACC = 5
    SAVE_ACC = 6

class PE_SUM_State(Enum):
    FREE = 0
    LOAD_RB_IN = 1
    LOAD_OP1 = 2
    LOAD_OP2 = 3
    SAVE_ACC = 4

class NPUInstruct:
    def __init__(self,A,B,C,source_selector,save_to_shared,save_to_buffout,
                 reset_vb,disable_clock,use_sm,we_vectorbuilder,sm_fastdata):
        self.A = A
        self.B = B
        self.C = C
        self.source_selector = source_selector
        self.save_to_shared = save_to_shared
        self.save_to_buffout = save_to_buffout
        self.reset_vb = reset_vb
        self.disable_clock = disable_clock
        self.use_sm = use_sm
        self.we_vectorbuilder = we_vectorbuilder
        self.sm_fastdata = sm_fastdata
    def __str__(self):
        instr = ''
        match(self.source_selector):
            case Source.SYSTOLIC_ARRAY:
                instr = (f'MUL A={self.A} B={self.B} C={self.C}')
            case Source.SIGMOID:
                instr = f'SIGMOID A={self.A} C={self.C}'
            case Source.VECTOR_BUILDER:
                instr = f'VB C={self.C}'
            case Source.SM:
                instr = f'LOAD B={self.B} C={self.C}'
            case Source.RB_IN:
                instr = f'LOAD_RB B={self.B} C={self.C}'
            case _:
                instr = 'UNK'
        return instr + (f' [{self.A}]->vector[{self.B}]' if self.we_vectorbuilder else '') +(' d_c' if self.disable_clock else '') + (f' ->SM[{self.B}]' if self.save_to_shared else '') + (' save_rbout' if self.save_to_buffout else '') + (' vector={0,0,0,0}' if self.reset_vb else '')
    def __int__(self):
        return self.A | \
            self.B << 10| \
            self.C << 20 | \
            self.source_selector.value << 30 | \
            self.save_to_shared << 34| \
            self.save_to_buffout << 35| \
            self.reset_vb << 36| \
            self.disable_clock << 37| \
            self.use_sm << 38| \
            self.we_vectorbuilder << 39| \
            self.sm_fastdata << 40
            
class Source(Enum):
    SYSTOLIC_ARRAY = 0
    SIGMOID = 1
    VECTOR_BUILDER = 2
    SM = 3
    RB_IN = 4
    SM_LONG_1 = 5
    SM_LONG_2 = 6
    SM_LONG_3 = 7
    SM_LONG_4 = 8

def get_first_free_pe(states):
    ind = list(filter(lambda x: x[1] == PE_State.FREE, enumerate(states)))
    if ind == []:
        return -1
    return ind[0][0]

#BUG: если на предыдущем такте был доступ в память, а в этом не ожидается
#     то в текущем будет скип
# l - размер вектора матрицы
def mul_col_col(addr1, addr2, l, sm_addr_out):
    l = l//4 + (1 if l%4 != 0 else 0)
    int_size = l//PEs
    occupies = [list(range(int_size*i,int_size*(i+1))) for i in range(PEs)] # ячейки, которые будут считать ядра
    int_size_1 = (l - int_size*PEs)
    for i in range(int_size_1):
        occupies[i] += list(range(int_size*PEs+i, int_size*PEs+i+1))
    part_sums = [alloc_lm(pe) for pe in range(PEs)]
    mul_bufs = [alloc_lm(pe) for pe in range(PEs)]
    operands_addrs = [[alloc_lm(i), alloc_lm(i)] for i in range(PEs)]
    steps = [0 for i in range(PEs)]
    states = [[PE_State.FREE,PE_MUL_State.FREE] for pe in range(PEs)]
    ind = 0
    instructs = [[] for pe in range(PEs)]
    for i in range(PEs):
        unused = get_first_free_addr_lm(i)
        instructs[i] += [NPUInstruct(unused, unused, unused, Source.SYSTOLIC_ARRAY, False, False, True, False, False, False, False)]
        instructs[i] += [NPUInstruct(unused, unused, unused, Source.SYSTOLIC_ARRAY, False, True, False, False, False, False, False)]
    def make_ld1(instructs, pe, sm_addr, lm_addr):
        # instructs[pe] += [('load', sm_addr, lm_addr)]
        instructs[pe] += [NPUInstruct(0, sm_addr, lm_addr, Source.SM, False, False, False, False, True, False, False)]
        states[pe][0] = PE_State.MEM
        states[pe][1] = PE_MUL_State.LOAD_OP1
    def make_ld2(instructs, pe, sm_addr, lm_addr):
        # instructs[pe] += [('load', sm_addr, lm_addr)]
        instructs[pe] += [NPUInstruct(0, sm_addr, lm_addr, Source.SM, False, False, False, False, True, False, False)]
        states[pe][0] = PE_State.MEM
        states[pe][1] = PE_MUL_State.LOAD_OP2
    def make_mul(instructs, pe, lm_addr1, lm_addr2, lm_dest):
        # instructs[pe] += [('mul', lm_addr1, lm_addr2, lm_dest)]
        instructs[pe] += [NPUInstruct(lm_addr1, lm_addr2, lm_dest, Source.SYSTOLIC_ARRAY, False, False, True, False, False, False, False)]
        states[pe][0] = PE_State.OCCUPIED
        states[pe][1] = PE_MUL_State.MUL
    def make_nop(instructs, pe):
        unused_addr = get_first_free_addr_lm(pe)
        # instructs[pe] += [('mul', unused_addr, unused_addr, unused_addr)]
        instructs[pe] += [NPUInstruct(unused_addr, unused_addr, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, False, False)]
        states[pe][0] = PE_State.FREE
    def make_sum(instructs, pe, lm_addr):
        # instructs[pe] += [('sum1', lm_addr, 0)]
        unused_addr = get_first_free_addr_lm(pe)
        instructs[pe] += [NPUInstruct(lm_addr, 0, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, True, False)]
        states[pe][0] = PE_State.OCCUPIED
        states[pe][1] = PE_MUL_State.SUM
    def make_acc(instructs, pe, lm_addr):
        # instructs[pe] += [('sum1', lm_addr, 1)]
        unused_addr = get_first_free_addr_lm(pe)
        instructs[pe] += [NPUInstruct(lm_addr, 1, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, True, False)]
        states[pe][0] = PE_State.OCCUPIED
        states[pe][1] = PE_MUL_State.ACC
    def make_save_acc(instructs, pe, lm_addr, rb_out):
        if rb_out:
            # instructs[pe] += [('sum4', lm_addr, 'rb_out')]
            instructs[pe] += [NPUInstruct(0, 1, lm_addr, Source.VECTOR_BUILDER, False, True, False, False, False, False, False)]
        else:
            # instructs[pe] += [('sum4', lm_addr)]
            instructs[pe] += [NPUInstruct(0, 1, lm_addr, Source.VECTOR_BUILDER, False, False, False, False, False, False, False)]
        states[pe][0] = PE_State.OCCUPIED
        states[pe][1] = PE_MUL_State.SAVE_ACC
    def is_mem_free(states):
        return all(map(lambda x: x != PE_State.MEM, states))
    while steps != [len(i) for i in occupies] or ind != 0:
        pe = ind
        if steps[pe] == len(occupies[pe]):
            make_nop(instructs, pe)
        else:
            if states[pe][1] == PE_MUL_State.FREE:
                addr1_pe = addr1 + occupies[pe][steps[pe]] # sm
                if is_mem_free(map(lambda x: x[0], states[:pe])):
                    make_ld1(instructs, pe, addr1_pe, operands_addrs[pe][0])
                else:
                    make_nop(instructs, pe)
            elif states[pe][1] == PE_MUL_State.LOAD_OP1:
                addr2_pe = addr2 + occupies[pe][steps[pe]] # sm
                if is_mem_free(map(lambda x: x[0], states[:pe])):
                    make_ld2(instructs, pe, addr2_pe, operands_addrs[pe][1])
                else:
                    make_nop(instructs, pe)
            elif states[pe][1] == PE_MUL_State.LOAD_OP2:
                make_mul(instructs, pe, operands_addrs[pe][0], operands_addrs[pe][1], mul_bufs[pe])
            elif states[pe][1] == PE_MUL_State.MUL:
                make_sum(instructs, pe, mul_bufs[pe])
            elif states[pe][1] == PE_MUL_State.SUM:
                if steps[pe] != 0:
                    make_acc(instructs, pe, part_sums[pe])
                else:
                    states[pe] = [PE_State.FREE, PE_MUL_State.ACC]
                    continue
            elif states[pe][1] == PE_MUL_State.ACC:
                make_save_acc(instructs, pe, part_sums[pe], steps[pe]+1 == len(occupies[pe]))
            elif states[pe][1] == PE_MUL_State.SAVE_ACC:
                steps[pe] += 1
                states[pe] = [PE_State.FREE, PE_MUL_State.FREE]
                continue
        ind = (ind+1)%PEs
    for pe, addr in enumerate(mul_bufs):
        free_lm(pe, addr)
    for pe, [addr1,addr2] in enumerate(operands_addrs):
        free_lm(pe, addr1)
        free_lm(pe, addr2)
    
    def reduce_sum(PEs_sources_count):
        pes_in_use = PEs_sources_count//2+PEs_sources_count%2
        occupies = [j+pes_in_use for j in range(pes_in_use)]

        received_sums = [alloc_lm(pe) for pe in range(pes_in_use)]
        instructs = [[] for pe in range(PEs)]
        def make_ld_rbin(instructs, pe, rb_in_ind, addr):
            # instructs[pe] += [('load_rbin', rb_in_ind if pe > rb_in_ind else rb_in_ind-1, addr)]
            instructs[pe] += [NPUInstruct(0, rb_in_ind if pe > rb_in_ind else rb_in_ind-1, addr, Source.RB_IN, False, False, True, False, False, False, False)]
        def make_load_op1(instructs, pe):
            # instructs[pe] += [('sum1', part_sums[pe], 0)]
            unused_addr = get_first_free_addr_lm(pe)
            instructs[pe] += [NPUInstruct(part_sums[pe], 0, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, True, False)]
        def make_load_op2(instructs, pe):
            # instructs[pe] += [('sum1', received_sums[pe], 1)]
            unused_addr = get_first_free_addr_lm(pe)
            instructs[pe] += [NPUInstruct(received_sums[pe], 1, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, True, False)]
        def make_nop(instructs, pe):
            addr = get_first_free_addr_lm(pe)
            # instructs[pe] += [('mul', addr, addr, addr)]
            instructs[pe] += [NPUInstruct(addr, addr, addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, False, False)]
        def make_save_acc(instructs, pe):
            # instructs[pe] += [('sum4', part_sums[pe], 'rb_out')]
            unused = get_first_free_addr_lm(pe)
            instructs[pe] += [NPUInstruct(unused, unused, part_sums[pe], Source.VECTOR_BUILDER, False, False, False, False, False, False, False)]
        state = PE_SUM_State.FREE
        pe = 0
        while state != PE_SUM_State.SAVE_ACC or pe != 0:
            if pe >= pes_in_use:
                make_nop(instructs, pe)
            else:
                if state == PE_SUM_State.FREE:
                    make_ld_rbin(instructs, pe, occupies[pe], received_sums[pe])
                    if pe == pes_in_use-1:
                        state = PE_SUM_State.LOAD_RB_IN
                elif state == PE_SUM_State.LOAD_RB_IN:
                    make_load_op1(instructs, pe)
                    if pe == pes_in_use-1:
                        state = PE_SUM_State.LOAD_OP1
                elif state == PE_SUM_State.LOAD_OP1:
                    make_load_op2(instructs, pe)
                    if pe == pes_in_use-1:
                        state = PE_SUM_State.LOAD_OP2
                elif state == PE_SUM_State.LOAD_OP2:
                    make_save_acc(instructs, pe)
                    if pe == pes_in_use-1:
                        state = PE_SUM_State.SAVE_ACC
            pe = (pe+1)%PEs
        for pe in range(pes_in_use,PEs_sources_count):
            free_lm(pe, part_sums[pe])
        if PEs_sources_count > 2:
            r = reduce_sum(pes_in_use)
            for i in range(PEs):
                instructs[i] += r[i]
        return instructs
    r = reduce_sum(PEs)
    for i in range(PEs):
        instructs[i] += r[i]
    # pe[0].lm[0] содержит вектор типа [num1, num2, 0, 0]
    # instructs[0] += [('sum1', part_sums[0], 0)]
    unused = get_first_free_addr_lm(0)
    instructs[0] += [NPUInstruct(part_sums[0], 0, unused, Source.SYSTOLIC_ARRAY, False, False, False, False, False, True, False)]
    for i in range(1, PEs):
        # instructs[i] += [('nop',)]
        unused = get_first_free_addr_lm(i)
        instructs[i] += [NPUInstruct(unused, unused, unused, Source.SYSTOLIC_ARRAY, False, False, False, False, False, False, False)]

    instructs[0] += [NPUInstruct(0, sm_addr_out, part_sums[0], Source.VECTOR_BUILDER, True, False, False, False, False, False, False)]
    
    for i in range(1, PEs):
        # instructs[i] += [('nop',)]
        unused = get_first_free_addr_lm(i)
        instructs[i] += [NPUInstruct(unused, unused, unused, Source.SYSTOLIC_ARRAY, False, False, False, False, False, False, False)]
    free_lm(0, part_sums[0])

    return instructs

class PE_SIGM_State(Enum):
    FREE = 0
    LOAD = 1
    SAVE = 2

def make_sigmoid(addr1, l, sm_addr_out):
    int_size = l//PEs
    occupies = [list(range(int_size*i, int_size*(i+1))) for i in range(PEs)]
    int_size_1 = (l - int_size*PEs)
    for i in range(int_size_1):
        occupies[i] += list(range(int_size*PEs+i, int_size*PEs+i+1))
    steps = [0 for i in range(PEs)]
    states = [[PE_State.FREE,PE_SIGM_State.FREE] for pe in range(PEs)]
    pe = 0
    instructs = [[] for pe in range(PEs)]
    results = [[alloc_lm(pe) for i in range(len(occupies[pe]))] for pe in range(PEs)]
    def is_mem_free(states):
        return all(map(lambda x: x != PE_State.MEM, states))
    def make_nop(instructs, pe):
        unused_addr = get_first_free_addr_lm(pe)
        # instructs[pe] += [('mul', unused_addr, unused_addr, unused_addr)]
        instructs[pe] += [NPUInstruct(unused_addr, unused_addr, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, False, False)]
        states[pe][0] = PE_State.FREE
    def make_ld(instructs, pe, sm_addr, lm_addr):
        instructs[pe] += [NPUInstruct(0, sm_addr, lm_addr, Source.SM, False, False, False, False, True, False, False)]
        states[pe][0] = PE_State.MEM
        states[pe][1] = PE_SIGM_State.LOAD
    def make_save(instructs, pe, lm_addr, sm_addr):
        instructs[pe] += [NPUInstruct(lm_addr, sm_addr, lm_addr, Source.SIGMOID, True, False, False, False, False, False, False)]
        states[pe][0] = PE_State.MEM
        states[pe][1] = PE_SIGM_State.SAVE
    while steps != [len(i) for i in occupies] or pe != 0:
        if steps[pe] == len(occupies[pe]):
            make_nop(instructs, pe)
        else:
            addr_pe = addr1 + occupies[pe][steps[pe]] # sm
            if states[pe][1] == PE_SIGM_State.FREE:
                if is_mem_free(map(lambda x: x[0], states[:pe])):
                    make_ld(instructs, pe, addr_pe, results[pe][steps[pe]])
                else:
                    make_nop(instructs, pe)
            elif states[pe][1] == PE_SIGM_State.LOAD:
                if is_mem_free(map(lambda x: x[0], states[:pe])):
                    make_save(instructs, pe, results[pe][steps[pe]], sm_addr_out+occupies[pe][steps[pe]])
                else:
                    make_nop(instructs, pe)
            elif states[pe][1] == PE_SIGM_State.SAVE:
                steps[pe] += 1
                states[pe] = [PE_State.FREE, PE_SIGM_State.FREE]
                continue
        pe = (pe+1)%PEs
    return instructs

class PE_COMPACT_State(Enum):
    FREE = 0
    LOAD = 1
    SAVE_TO_VECTOR = 2
    SAVE_TO_SHARED = 3


# преобразует массивы вида [num1, 0, 0, 0], [num2, 0, 0, 0], [num3, 0, 0, 0]
# в [num1, num2, num3, 0]
# l - количество векторов размера 4 ПЕРЕД упаковкой (для примера выше будет равен 3)
def make_compact(addr, l, out_addr):
    l = l//4 + (1 if l%4 != 0 else 0)
    int_size = l//PEs
    occupies = [list(range(int_size*i, int_size*(i+1))) for i in range(PEs)]
    int_size_1 = (l - int_size*PEs)
    for i in range(int_size_1):
        occupies[i] += list(range(int_size*PEs+i, int_size*PEs+i+1))
    steps = [0 for i in range(PEs)]
    sub_steps = [0 for i in range(PEs)]
    states = [[PE_State.FREE,PE_COMPACT_State.FREE] for pe in range(PEs)]
    pe = 0
    instructs = [[] for pe in range(PEs)]
    results = [[alloc_lm(pe) for i in range(len(occupies[pe]))] for pe in range(PEs)]
    def is_mem_free(states):
        return all(map(lambda x: x != PE_State.MEM, states))
    def make_nop(instructs, pe):
        unused_addr = get_first_free_addr_lm(pe)
        # instructs[pe] += [('mul', unused_addr, unused_addr, unused_addr)]
        instructs[pe] += [NPUInstruct(unused_addr, unused_addr, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, False, False)]
        states[pe][0] = PE_State.FREE
    def make_load(instructs, pe, sm_addr, lm_addr, clear_vector):
        instructs[pe] += [NPUInstruct(0, sm_addr, lm_addr, Source.SM, False, False, clear_vector, False, True, False, False)]
        states[pe][0] = PE_State.MEM
        states[pe][1] = PE_COMPACT_State.LOAD
    def make_save_to_vector(instructs, pe, lm_addr, v_ind):
        unused_addr = get_first_free_addr_lm(pe)
        instructs[pe] += [NPUInstruct(lm_addr, v_ind, unused_addr, Source.SYSTOLIC_ARRAY, False, False, False, False, False, True, False)]
        states[pe][0] = PE_State.OCCUPIED
        states[pe][1] = PE_COMPACT_State.SAVE_TO_VECTOR
    def make_save(instructs, pe, sm_addr):
        unused = get_first_free_addr_lm(pe)
        instructs[pe] += [NPUInstruct(unused, sm_addr, unused, Source.VECTOR_BUILDER, True, False, False, False, False, False, False)]
        states[pe][0] = PE_State.MEM
        states[pe][1] = PE_COMPACT_State.SAVE_TO_SHARED
    while steps != [len(i) for i in occupies] or pe != 0:
        if steps[pe] == len(occupies[pe]):
            make_nop(instructs, pe)
        else:
            addr_pe = addr + occupies[pe][steps[pe]]*4+sub_steps[pe]
            if states[pe][1] == PE_COMPACT_State.FREE:
                if is_mem_free(map(lambda x:x[0],states[:pe])):
                    make_load(instructs, pe, addr_pe, results[pe][steps[pe]], sub_steps[pe]==0)
                else:
                    make_nop(instructs, pe)
            elif states[pe][1] == PE_COMPACT_State.LOAD:
                make_save_to_vector(instructs, pe, results[pe][steps[pe]], sub_steps[pe])
                sub_steps[pe]+=1
            elif states[pe][1] == PE_COMPACT_State.SAVE_TO_VECTOR:
                if sub_steps[pe] == 4:
                    sub_steps[pe] = 0
                    make_save(instructs, pe, out_addr+occupies[pe][steps[pe]])
                else:
                    states[pe] = [PE_State.FREE, PE_COMPACT_State.FREE]
                    continue
            elif states[pe][1] == PE_COMPACT_State.SAVE_TO_SHARED:
                states[pe] = [PE_State.FREE, PE_COMPACT_State.FREE]
                steps[pe] += 1
                continue
        pe = (pe+1)%PEs
    for pe, addrs in enumerate(results):
        for addr in addrs:
            free_lm(pe, addr)
    return instructs
#
# тесты функций
#
# inst = make_compact(0, 25, 0)

# inst = mul_col_col(layers_indx_sm[0][0][0], layers_indx_sm[1][0][0], 2, 100)
# print([len(inst[i]) for i in range(PEs)])

# inst = make_sigmoid(10, 4, 100)

# for i in range(len(inst[0])):
#     print([f'{str(inst[j][i]):30}' for j in range(PEs)])
# exit(0)


#
# сборка модулей
#
instructions = [[] for i in range(PEs)]
muls_inds = []
prev_out_addr = -1
for ind, layer in enumerate(model):
    if layer[0] == 'in':
        prev_out_addr = layers_indx_sm[ind][0][0]
    elif layer[0] == 'sig':
        l = len(layer[1][0])
        l = l//4 + (1 if l %4 != 0 else 0)
        inst = make_sigmoid(prev_out_addr, l, layers_indx_sm[ind][0][0])
        prev_out_addr = layers_indx_sm[ind][0][0]
        for pe in range(PEs):
            instructions[pe] += inst[pe]
    elif layer[0] == 'mul':
        ll = len(layer[1]) # количество столбцов матрицы
        l = len(layer[1][0]) # количество строк матрицы
        out_inds = [sm_ind + i for i in range(ll)]
        sm_ind += ll
        muls_inds += [(0,out_inds)]
        
        for l2 in range(ll):
            inst = mul_col_col(prev_out_addr, layers_indx_sm[ind][l2][0], l, out_inds[l2])
            for pe in range(PEs):
                instructions[pe] += inst[pe]
        out_inds1 = [sm_ind+i for i in range(ll//4 + (1 if ll%4 != 0 else 0))]
        muls_inds += [(1,out_inds1)]
        sm_ind += len(out_inds1)
        inst = make_compact(out_inds[0], ll, out_inds1[0])
        for pe in range(PEs):
            instructions[pe] += inst[pe]
        prev_out_addr = out_inds1[0]
for i in range(PEs):
    instructions[i][-1].disable_clock=True

out_addr, out_len = prev_out_addr, len(layers_indx_sm[-1])//4 + (1 if len(layers_indx_sm[-1]) %4 != 0 else 0)
print(f'out_addr={out_addr}, len={out_len}')

# for i in range(100,len(instructions[0])):
#     print(i+1,[str(instructions[pe][i]) for pe in range(PEs)])
# print([len(instructions[i]) for i in range(PEs)])

with open('gen.sv','w') as f:
    print(f'inst_count = {len(instructions[0])};', file=f)
    print(f'out_addr = {out_addr};', file=f)
    print(f'out_len = {out_len};', file=f)
    sm_prefix = 'm.shared_mem.ram'
    for i1 in layers_indx_sm:
        for i2 in i1:
            for i3 in i2:
                print(sm_prefix+f'[{i3}] = 32\'h{hex(sum([sm_layout[i3][j]<<(8*j) for j in range(4)]))[2:]};', file=f)
    pes_prefix = 'm.pe_inst'
    for pe in range(PEs):
        for ind, i in enumerate(instructions[pe]):
            print(pes_prefix+f'[{pe}].pe_core.instruction_mem[{ind+1}] = 41\'h{hex(int(i))[2:]};', file=f)
        print(file=f)
