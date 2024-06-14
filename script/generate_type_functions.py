#!/usr/bin/python

import re
import shutil

TYPE = "Type"

DIR = "../src/base"

type_file = open(DIR+"/type.h", "r")
type_file_content = type_file.read();
type_file.close()

tmp = type_file_content[type_file_content.find(TYPE):]
type_content  = tmp[tmp.find("{")+1:tmp.find("}")]
#print "Before:"
# print (type_file_content)

type_content_without_comment = re.sub("/\*((?:.|\n)*?)\*/", "", type_content)
type_content_without_comment = re.sub("//.*\n", "", type_content_without_comment)
type_content_tmp = re.sub("\s*", "", type_content_without_comment)
type_content_main = re.sub("=\d*", "", type_content_tmp).split(',')
type_names = type_content_main[:type_content_main.index("TYPESIZE")]
first_synapse_idx = type_names.index('Static')

#print "After:"
#print type_names

func_file = open(DIR+"/TypeFunc.h", "r")
func_file_content = func_file.readlines()
func_file.close()

func_mains = []
func_names = []

for line in func_file_content:
    line = line.strip()
    if line.startswith("extern"):
        line = line[line.find("extern")+6:line.find(";")].strip()
        if not line.startswith("BlockSize"):
            get_name = re.search("\(\*(.*)\[", line)
            if get_name:
                func_names.append(get_name.group(1))
                func_mains.append(line.replace("TYPESIZE", ""))
            else:
                print(line)
#print func_names
#print func_mains

#shutil.copyfile(DIR+"/TypeFunc.cu", DIR+"/TypeFunc.cu.copied")
#shutil.copyfile(DIR+"/TypeFunc.cpp", DIR+"/TypeFunc.cpp.copied")
#shutil.copyfile("../src/mpi_utils/mpi_func.cpp", "../src/mpi_utils/mpi_func.cpp.copied")

cu_file = open(DIR+"/TypeFunc.cu", "w")
cpp_file = open(DIR+"/TypeFunc.cpp", "w")

warn_info = "/* This file is generated by scripts automatively.\n * do not change it by hand.\n */\n\n"
common_headers = '#include "../utils/template.h"\n#include "../base/TypeFunc.h"\n\n'

# FIXME: 目前只检查是否包含 LIF 和 IAF 神经元，后续需要修改
is_neuron = lambda name: name.find("LIF") >= 0 or name.find("IAF") >= 0

for i,t in enumerate(type_names):
    type_ = "neuron" if is_neuron(t) else "synapse"
    # if i >= first_synapse_idx: 
    #     type_ = 'synapse'
    # ! 文件夹一律按 TypeName.lower() 命名，所有源代码放在下面，不要有二级目录
    common_headers += '#include "../{}/{}/{}Data.h"\n'.format(type_, t.lower(), t)

common_headers += '\n'

cu_content = []
cpp_content = []
mpi_content = []

cu_content.append(warn_info + common_headers)
cpp_content.append(warn_info + common_headers)

for (name, body) in zip(func_names, func_mains):
    tmp = body + " = {"

    # if name.find("Find") >= 0:
    #     paras = re.search("\(\*.*\\)\((.*)\)", body).group(1)
    #     for type_name in type_names:
    #         if type_names.index(type_name) != type_names.index("LIF"):
    #             tmp += (" nullFunc<{}>,".format(paras))
    #         else:
    #             tmp += (" " + name.replace("Type", type_name) + ",")

    if name.find("AllType") >= 0:
            for type_name in type_names:
                if type_names.index(type_name) >=  type_names.index("LIF"):
                    tmp += ("\n\t" + name.replace("Type", type_name) + ",")
                else:
                    tmp += ("\n\t" + name.replace("AllType", type_name) + ",")
    elif name.find("Connection") >= 0:
        for type_name in type_names:
            if is_neuron(type_name):
            # if type_names.index(type_name) < type_names.index("Static"):
                tmp += ("\n\tNULL,")
            else:
                tmp += ("\n\t" + name.replace("Type", type_name) + ",")
    elif name.find("Neuron") >= 0:
        for type_name in type_names:
            if is_neuron(type_name):
            # if type_names.index(type_name) < type_names.index("Static"):
                tmp += ("\n\t" + name.replace("Neuron", type_name) + ",")
            else:
                tmp += ("\n\tNULL,")
    elif name.find("Synapse") >= 0:
        for type_name in type_names:
            if is_neuron(type_name):
            # if type_names.index(type_name) < type_names.index("Static"):
                tmp += ("\n\tNULL,")
            else:
                tmp += ("\n\t" + name.replace("Synapse", type_name) + ",")
    else:
        for type_name in type_names:
            tmp += ("\n\t" + name.replace("Type", type_name) + ",")
        
    tmp = tmp[:-1] + "\n};\n\n"

    # print(tmp)

    if name.startswith("cuda"):
        cu_content.append(tmp)
    elif name.startswith("mpi"):
        mpi_content.append(tmp)
    else:
        cpp_content.append(tmp)

cu_file.writelines(cu_content)
cpp_file.writelines(cpp_content)
cpp_file.writelines(mpi_content)
#print mpi_content

cu_file.close()
cpp_file.close()

