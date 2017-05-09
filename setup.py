from distutils.core import setup, Extension

pyadsb3 = Extension('_pic2pic',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y', '-g'], 
        libraries = ['boost_python', 'opencv_imgproc', 'opencv_core'],
        include_dirs = ['/usr/local/include'],
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp', 'colorize.cpp']
        )

setup (name = 'pyadsb3',
       version = '0.0.1',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [pyadsb3],
       )
