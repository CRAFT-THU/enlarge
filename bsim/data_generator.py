
from generator import *

C_TYPE_SORT = {
    'char' : 0,
    'unsigned char' : 1,
    'short' : 2,
    'unsigned short' : 3,
    'int' : 4,
    'unsigned int' : 5,
    'float' : 6,
    'long' : 7,
    'unsigned long': 8,
    'long long' : 9,
    'unsigned long long': 10,
    'double' : 11,
    'long double' : 12
}

def mycap(v):
    return v[0].upper() + v[1:]

def myhash(v):
    if v in C_TYPE_SORT:
        return C_TYPE_SORT[v]
    else:
        return len(C_TYPE_SORT) + abs(hash(v))

class Data(object):
    def __init__(self, name, parameters, path='./', pre='', post='Data',
            headers=[], cu_headers=[]):
        self.name = mycap(name);
        self.classname = "{}{}{}".format(pre, name, post)
        self.path = path
        self.headers = headers
        self.cu_headers = cu_headers
        self.parameters = {k:parameters[k] for k in sorted(parameters.keys(), key= lambda x:myhash(x), reverse = False)}

    def generate_h(self):
        h = HGenerator("{}/{}.h".format(self.path, self.classname))

        h.include_std("stdio.h")
        h.blank_line()
        for i in self.headers:
            h.include(i)
        # h.include("../../utils/macros.h")
        h.blank_line()

        h.struct(self.classname, 0)
        for k in self.parameters:
            for v in self.parameters[k]:
                h.line("{} *p{}".format(k, mycap(v)))
            h.blank_line()
        h.struct_end()
        h.blank_line()

        # h.line("DATA_FUNC_DEFINE({})".format(self.name), 0)
        
        h.blank_line()
        h.func("void *malloc{}()".format(self.name))
        h.func("void *alloc{}(int num)".format(self.name))
        h.func("int free{}(void *pCPU, int num)".format(self.name))
        h.func("int alloc{}Para(void *pCPU, int num)".format(self.name))
        h.func("int free{}Para(void *pCPU, int num)".format(self.name))
        h.func("int save{}(void *pCPU, int num, FILE *f)".format(self.name))
        h.func("int load{}(int num, FILE *f)".format(self.name))
        h.blank_line()

        h.func("void *cudaAlloc{}(void *pCPU, int num)".format(self.name))
        h.func("int cuda{}ToGPU(void *pCPU, void *pGPU, int num)".format(self.name))
        h.func("void cudaUpdate{}(void *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int start_id, int t, BlockSize *pSize)".format(self.name))
        h.func("int cudaFree{}(void *pGPU)".format(self.name))
        h.blank_line()

        h.func("int mpiSend{}(void *data, int rank, int offset, int size)".format(self.name))
        h.func("int mpiRecv{}(void *data, int rank, int offset, int size)".format(self.name))
        h.blank_line()

        h.close()
        return 0

    def generate_c(self):
        c = CGenerator("{}/{}.cpp".format(self.path, self.classname))
        c.include_std("stdlib.h")
        c.include_std("string.h")
        c.blank_line()
        c.include("{}.h".format(self.classname))
        c.blank_line()

        c.func_start("void *malloc{}()".format(self.name))
        c.malloc("p", "{}".format(self.classname), 1)
        c.func_end("(void*)p")
        c.blank_line()

        c.func_start("void *alloc{}(int num)".format(self.name))
        c.line("void *p = malloc{}()".format(self.name))
        c.line("alloc{}Para(p, num)".format(self.name, self.name, self.name))
        c.func_end("p")
        c.blank_line()

        c.func_start("int free{}(void *pCPU, int num)".format(self.name))
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("free(p->p{})".format(mycap(p)))
            c.blank_line()
        c.line("free(p)")
        c.func_end("0")
        c.blank_line()

        c.func_start("int alloc{}Para(void *pCPU, int num)".format(self.name))
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("p->p{} = ({}*)malloc(n*sizeof({}))".format(mycap(p), t, t))
            c.blank_line()
        c.func_end(0)
        c.blank_line()

        c.func_start("int free{}Para(void *pCPU, int num)".format(self.name))
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("free(p->p{})".format(mycap(p)))
            c.blank_line()
        c.func_end("0")
        c.blank_line()

        c.func_start("int save{}(void *pCPU, int num, FILE *f)".format(self.name))
        c.blank_line()
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("fwrite(p->p{}, sizeof({}), num, f)".format(mycap(p), t))
            c.blank_line()
        c.func_end("0")
        c.blank_line()

        c.func_start("int load{}(int num, FILE *f)".format(self.name))
        c.line("{} *p = ({}*)malloc(sizeof({}))".format(self.classname, self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("fread(p->p{}, sizeof({}), num, f)".format(mycap(p), t))
            c.blank_line()
        c.func_end("p")
        c.blank_line()

        return 0

    def generate_cu(self):
        cu = CUDAGenerator("{}/{}.cu".format(self.path, self.classname)) 
        cu.include_std("stdlib.h")
        cu.include_std("string.h")
        for i in self.cu_headers:
            cu.include(i)
        cu.include("{}.h".format(self.classname))
        cu.blank_line()

        cu.func_start("void *cudaAlloc{}(void *pCPU, int num)".format(self.name))
        cu.line("void *ret = NULL")
        cu.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        cu.malloc("tmp", "{}".format(self.classname), 1)
        cu.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                cu.to_gpu("tmp->p{}".format(mycap(p)),
                          cpu="p->p{}".format(mycap(p)),
                          type_=t, num="num");
            cu.blank_line()
        cu.to_gpu("ret", cpu="tmp", type_="{}".format(self.classname), num=1)
        cu.free("tmp")
        cu.func_end("ret")
        cu.blank_line()

        cu.func_start("void *cuda{}ToGPU(void *pCPU, void *pGPU, int num)".format(self.name))
        cu.line("{} *pC = ({}*)pCPU".format(self.classname, self.classname))
        cu.line("{}Data *pG = ({}Data*)pGPU".format(self.name, self.name))
        cu.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                cu.cpu_to_gpu(cpu="pC", gpu="pG", type_=t, num="num")
            cu.blank_line()
        cu.func_end(0)
        cu.blank_line()

        cu.func_start("void *cudaFree{}(void *pGPU)".format(self.name))
        cu.free_gpu("pGPU")
        cu.func_end(0)
        cu.blank_line()

        cu.close()
        return 0

    def generate_mpi(self):
        c = CGenerator("{}/{}.cpp".format(self.path, self.classname))
        c.include("mpi.h")
        c.blank_line()
        c.include("{}.h".format(self.classname))
        c.blank_line()

        c.func_start("void *malloc{}()".format(self.name))
        c.line("void *p = malloc(sizeof({}))".format(self.classname))
        c.func_end("p")
        c.blank_line()

        c.func_start("void *alloc{}(int num)".format(self.name))
        c.line("void *p = malloc{}()".format(self.name, self.name, self.name))
        c.line("alloc{}Para(p, num)".format(self.name, self.name, self.name))
        c.func_end("p")
        c.blank_line()

        c.func_start("int free{}(void *pCPU, int num)".format(self.name))
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("free(p->p{})".format(mycap(p)))
            c.blank_line()
        c.line("free(p)")
        c.func_end("0")
        c.blank_line()

        c.func_start("int alloc{}Para(void *pCPU, int num)".format(self.name))
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("p->p{} = ({}*)malloc(n*sizeof({}))".format(mycap(p), t, t))
            c.blank_line()
        c.func_end(0)
        c.blank_line()

        c.func_start("int free{}Para(void *pCPU, int num)".format(self.name))
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("free(p->p{})".format(mycap(p)))
            c.blank_line()
        c.func_end("0")
        c.blank_line()

        c.func_start("int save{}(void *pCPU, int num, FILE *f)".format(self.name))
        c.blank_line()
        c.line("{} *p = ({}*)pCPU".format(self.classname, self.classname))
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("fwrite(p->p{}, sizeof({}), num, f)".format(mycap(p), t))
            c.blank_line()
        c.func_end("0")
        c.blank_line()

        c.func_start("int load{}(int num, FILE *f)".format(self.name))
        c.line("{} *p = ({}*)malloc(sizeof({}))".format(self.classname,
            self.classname, self.classname))
        c.blank_line()
        for t in self.parameters:
            for p in self.parameters[t]:
                c.line("fread(p->p{}, sizeof({}), num, f)".format(mycap(p), t))
            c.blank_line()
        c.func_end("p")
        c.blank_line()

        return 0

if __name__ == '__main__':
    parameters_old = {
                    'refracStep' : 'int',
                    'refracTime' : 'int',
                    'vm' : 'real',
                    'ci' : 'real',
                    'ce' : 'real',
                    'c_i' : 'real',
                    'c_e' : 'real',
                    'v_tmp' : 'real',
                    'v_i' : 'real',
                    'v_e' : 'real',
                    'v_thresh' : 'real',
                    'v_reset' : 'real',
                    'v_m' : 'real',
                }
    parameters = {}
    for k,v in parameters_old.items():
        parameters.setdefault(v, []).append(k)

    lif = Data('LIF', parameters, 
            path='../src/neuron/lif/', pre='G', post='Neurons', 
            headers=['../../utils/type.h', '../../utils/BlockSize.h'], 
            cu_headers=['../../third_party/cuda/helper_cuda.h'])
    lif.generate_h()
    lif.generate_c()
    lif.generate_cu()

    static = Data('Static', {'int':['dst'], 'real':['weight']},
            path='../src/synapse/static/', pre='G', post='Synapses',
            headers=['../../utils/type.h', '../../utils/BlockSize.h'], 
            cu_headers=['../../third_party/cuda/helper_cuda.h'])
    static.generate_h()
    static.generate_c()
    static.generate_cu()
