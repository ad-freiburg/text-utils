## Text correction utilities

This repository contains Python utilities (backed by Rust) used in many Deep 
Learning based NLP applications.

It is mainly designed to be used for whitespace correction, 
spelling correction, and similar tasks.

You can install the Python package from PyPi via

> pip install text-correction-utils

However, we encourage you to use the native installation method. This way the 
Rust code is compiled for your particular CPU. Installation steps:
1. Clone the repository
   > git clone https://github.com/bastiscode/text-correction-utils.git
2. Install natively
   > make -C text-correction-utils build_native
