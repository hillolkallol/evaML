# ===============================================================================
# MIT License
#
# Copyright (c) 2021 Kallol Das
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===============================================================================
# Authors: Kallol Das <kalloldash@gmail.com>
# ===============================================================================


from evaml.classification import KNearestNeighbors

def _generate_params_test():
    knn = KNearestNeighbors()
    print(knn._generate_params([12,13,14], [1,2,3], [12,13,14], [1,2,3]))


_generate_params_test()


# from numba import njit
# import numpy as np
#
# class KNearestNeighborsTest():
#     def __init__(self):
#         pass
#
#     @staticmethod
#     @njit
#     def test():
#         for a in np.zeros(10):
#             print(a)
#
# obj = KNearestNeighborsTest()
# obj.test()


test_dict = {
           "param-set-578":{
              "params":{
                 "neighbors":15,
                 "weight":"uniform",
                 "algorithm":"auto",
                 "leaf-size":20,
                 "p":2
              },
              "results":{
                 "precision":0.87,
                 "recall":0.87,
                 "f-score":0.86,
                 "accuracy":0.86
              },
              "learning_curve_plot_name":"learning_curve_15_uniform_auto_20_2"
           },
           "param-set-579":{
              "params":{
                 "neighbors":15,
                 "weight":"uniform",
                 "algorithm":"auto",
                 "leaf-size":20,
                 "p":3
              },
              "results":{
                 "precision":0.87,
                 "recall":0.87,
                 "f-score":0.86,
                 "accuracy":0.92
              },
              "learning_curve_plot_name":"learning_curve_15_uniform_auto_20_3"
           },
           "param-set-581":{
              "params":{
                 "neighbors":15,
                 "weight":"uniform",
                 "algorithm":"auto",
                 "leaf-size":21,
                 "p":2
              },
              "results":{
                 "precision":0.87,
                 "recall":0.87,
                 "f-score":0.86,
                 "accuracy":0.87
              },
              "learning_curve_plot_name":"learning_curve_15_uniform_auto_21_2"
           }
        }