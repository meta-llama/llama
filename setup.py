# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="llama",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
#include <iostream>
using namespace std;

int main() {
    int a = 5, b = 3, sum;
    sum = a + b;
    cout << "Sum of " << a << " and " << b << " is: " << sum << endl;
    return 0;
}
