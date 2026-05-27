import numpy
import os, sys, os.path, tempfile, subprocess, shutil
from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
from distutils.extension import Extension

from setuptools import setup, Extension

def checkOpenmpSupport():
    """ Adapted from https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    """ 
    ompTest = \
    r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Thread %d, Total number of threads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(ompTest)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)

    os.chdir(curdir)
    shutil.rmtree(tmpdir) 
    if result == 0:
        return True
    else:
        return False

if checkOpenmpSupport() == True:
    ompArgs = ['-fopenmp']
else:
    ompArgs = None 

shared_folder = os.path.abspath("/home/kingshuk/PhD/chiral-active-matter/pystokes2/pystokes2/") 

ext_modules = [
    Extension(
        name="UnboundedFilaments",  # The name of the compiled module
        sources=["UnboundedFilaments.pyx"],  # Your main .pyx file
        include_dirs=[
            numpy.get_include(),
            shared_folder  # Make sure the folder containing first.pxd is included
        ],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': sys.version_info[0],
            'linetrace': True
        }
    ),
    libraries=[],
)
