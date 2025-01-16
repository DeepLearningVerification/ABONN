import nnverify
import nnverify.proof_transfer.proof_transfer as pt
from nnverify.analyzer import Analyzer
from nnverify.bnb import bnb, Split, is_relu_split
import nnverify.specs.spec as specs
from verifier_util import Result_Olive, Results_Olive
from nnverify.common import Status
from nnverify import config
from nnverify.domains.deepz import ZonoTransformer
import verifier_util
import torch
import time
import pandas as pd
import json
import sys
import nnverify.proof_transfer.approximate as ap
from nnverify.analyzer import Analyzer
from nnverify.proof_transfer.pt_util import result_resolved, plot_verification_results
import argparse
from nnverify.bnb.proof_tree import ProofTree
import nnverify.attack
from nnverify.common import Status
# from dsplot.graph import Graph
from nnverify.domains.deepz import ZonoTransformer
from unittest import TestCase
from nnverify import config
from nnverify.bnb import Split
from nnverify.common import Domain
from nnverify.proof_transfer.param_tune import write_result
from nnverify.proof_transfer.pt_types import ProofTransferMethod, IVAN, REORDERING
from nnverify.common.dataset import Dataset
from nnverify.bnb import Split, is_relu_split, is_input_split
from nnverify.bnb import Split, is_relu_split, branch
import copy
from nnverify import common, config
import os
import numpy as np
import csv
from time import gmtime, strftime
import math
import pickle
import time
from nnverify.common.result import Result, Results
from nnverify.common import RESULT_DIR, strip_name
from verifier_util import Spec_D
from nnverify.training.training_args import TrainArgs



# Naiive breath-first TreeGrowing
class AnalyzerBase(Analyzer):
    @classmethod
    def class_name(cls):
        return cls.__name__

    def sub_class_name(self):
        if self.class_name( )== "Analyzer_Reuse" and len(self.template_store.template_map)== 0:
            return "Template_Gen"
        return self.class_name()

    def analyze_domain(self, props):
        results = Results_Olive(self.args, props = props, option=self.sub_class_name())
        for i in range(len(props)):
            print("************************** Proof %d *****************************" % ( i +1))
            num_clauses = props[i].get_input_clause_count()
            clause_ver_status = []
            ver_start_time = time.time()

            for j in range(num_clauses):
                cl_status, tree_size, visited_nodes, lb = self.analyze(props[i].get_input_clause(j))
                clause_ver_status.append(cl_status)
            status = self.extract_status(clause_ver_status)
            print(status)
            ver_time = time.time() - ver_start_time
            results.add_result(Result_Olive(ver_time, status, tree_size=tree_size, visited_nodes = visited_nodes, lb = lb))
        return results
    
    def serialize_sets(self,obj):
        if isinstance(obj, set):
            return list(obj)
        return obj
    
    def run_analyzer(self, index, eps=1/255, mode = "easy"):
        print('Using %s abstract domain' % self.args.domain)
        index = int(index)
        eps = float(eps)
        props, inputs = verifier_util.get_specs(self.args.dataset, spec_type=self.args.spec_type, count=self.args.count, eps=eps, mode = mode)
        props = [props[index]]
        results = self.analyze_domain(props)
        results.compute_stats()
        print('Results: ', results.output_count)
        print('Average time:', results.avg_time)
        return results
    
    def analyze(self, prop):
        self.update_transformer(prop)
        tree_size = 1
        node_visited= 1
        
        # Check if classified correctly
        if nnverify.attack.check_adversarial(prop.input, self.net, prop):
            return Status.MISS_CLASSIFIED, tree_size, node_visited, None

        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if nnverify.attack.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size, node_visited, None

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            bnb_analyzer = BnBBase(self.net, self.transformer, prop, self.args, self.template_store)
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()

            status = bnb_analyzer.global_status
            tree_size = bnb_analyzer.tree_size
            node_visited = bnb_analyzer.node_visited
            lb = bnb_analyzer.cur_lb
        return status, tree_size, node_visited, lb


class BnBBase(bnb.BnB):
    def __init__(self, net, transformer, init_prop, args, template_store, print_result=False):
        self.node_visited = 0
        super().__init__(net, transformer, init_prop, args, template_store, print_result)
    
    
    @classmethod
    def class_name(cls):
        return cls.__name__


    def sub_class_name(self):
        if self.class_name( )== "BaB_Reuse" and len(self.template_store.template_map)== 0:
            return "Template_Gen_BaB"
        return self.class_name()
    
    
    def store_final_tree(self):
        self.proof_tree = ProofTree(self.root_spec)
        self.template_store.add_tree(self.init_prop, self.proof_tree)
        # formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        # with open(f'./pickle/tree_{self.sub_class_name()}_{formatted_time}.pkl', 'wb') as file:
        #     # Pickle the variable and write it to the file
        #     pickle.dump(self.template_store, file)
    
    def verify_specs(self):
        for spec in self.cur_specs:
            self.update_transformer(spec.input_spec, relu_spec=spec.relu_spec)

            # Transformer is updated with new mask
            status, lb = self.verify_node(self.transformer, spec.input_spec)
            self.update_cur_lb(lb)
            spec.update_status(status, lb)
            print("\nSpecNode \t Depth: ", spec.depth, "LB: ", lb)
            if status == Status.ADV_EXAMPLE:
                self.global_status = status
                self.store_final_tree()
                if status == Status.ADV_EXAMPLE:
                    print(f"BnB baseline Finished Verified Specicications, visited nodes: {self.node_visited}, find a counterexample")
                    # with open("file.pickle", "wb") as handle:
                    #     pickle.dump(self.template_store, handle)
                    return
            if self.is_timeout():
                print(f"Time is up, visited nodes: {self.node_visited}")
                self.store_final_tree()
                self.check_verified_status()
                return

    def verify_node(self, transformer, prop):
        """
        It is called from bnb_relu_complete. Attempts to verify (ilb, iub), there are three possible outcomes that
        are indicated by the status: 1) verified 2) adversarial example is found 3) Unknown
        """
        lb, is_feasible, adv_ex = transformer.compute_lb(complete=True)
        status = self.get_status(adv_ex, is_feasible, lb)
        if lb is not None:
            lb = float(lb)
        self.node_visited += 1
        return status, lb

    def create_initial_specs(self, prop, unstable_relus):
        if is_relu_split(self.split):
            relu_spec = specs.create_relu_spec(unstable_relus)
            self.root_spec = Spec_D(prop, relu_spec=relu_spec, status=self.global_status)
            cur_specs = specs.SpecList([self.root_spec])
            config.write_log("Unstable relus: " + str(unstable_relus))
        else:
            if self.args.initial_split:
                # Do a smarter initial split similar to ERAN
                # This works only for ACAS-XU
                zono_transformer = ZonoTransformer(prop, complete=True)
                zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)

                center = zono_transformer.centers[-1]
                cof = zono_transformer.cofs[-1]
                cof_abs = torch.sum(torch.abs(cof), dim=0)
                lb = center - cof_abs
                adv_index = torch.argmin(lb)
                input_len = len(prop.input_lb)
                smears = torch.abs(cof[:input_len, adv_index])
                split_multiple = 10 / torch.sum(smears)  # Dividing the initial splits in the proportion of above score
                num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]

                inp_specs = prop.multiple_splits(num_splits)
                cur_specs = specs.SpecList([Spec_D(prop, status=self.global_status) for prop in inp_specs])
                # TODO: Add a root spec in this case as well
            else:
                self.root_spec = Spec_D(prop, status=self.global_status)
                cur_specs = specs.SpecList([self.root_spec])

        return cur_specs

    def run(self):

        """
        It is the public method called from the analyzer. @param split is a string that chooses the mode for relu
        or input splitting.
        """
        if self.global_status != Status.UNKNOWN:
            return

        while self.continue_search():

            self.prev_lb = self.cur_lb
            self.reset_cur_lb()

            # Main verification loop
            if self.args.parallel:
                self.verify_specs_parallel()
            else:
                self.verify_specs()

            split_score = self.set_split_score(self.init_prop, self.cur_specs, inp_template=self.inp_template)
            # Each spec should hold the prev lb and current lb
            self.cur_specs, verified_specs = verifier_util.branch_unsolved(self.cur_specs, self.split, split_score=split_score,
                                                                  inp_template=self.inp_template, args=self.args,
                                                                  net=self.net, transformer=self.transformer)
            # Update the tree size
            self.tree_size += len(self.cur_specs)

        print(f"BnB Baseline Finished Verified Specicications, visited nodes: {self.node_visited}")
        self.check_verified_status()
        self.store_final_tree()

    def get_unstable_relus(self):
        lb, is_feasible, adv_ex = self.transformer.compute_lb(complete=True)
        status = self.get_status(adv_ex, is_feasible, lb)

        if 'unstable_relus' in dir(self.transformer):
            unstable_relus = self.transformer.unstable_relus
        else:
            unstable_relus = None

        if status != Status.UNKNOWN:
            self.global_status = status
            if status == Status.VERIFIED and self.print_result:
                print(status)
        return unstable_relus
    
class Analyzer_MCTS(AnalyzerBase):
    def analyze(self, prop):
        self.update_transformer(prop)
        tree_size = 1
        node_visited= 1
        
        # Check if classified correctly
        if nnverify.attack.check_adversarial(prop.input, self.net, prop):
            return Status.MISS_CLASSIFIED, tree_size, node_visited, None

        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if nnverify.attack.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size, node_visited, None

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            bnb_analyzer = BnB_MCTS(self.net, self.transformer, prop, self.args, self.template_store)
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()

            status = bnb_analyzer.global_status
            tree_size = bnb_analyzer.tree_size
            node_visited = bnb_analyzer.node_visited
            lb = bnb_analyzer.cur_lb
        return status, tree_size, node_visited, lb
        
class BnB_MCTS(BnBBase):
    def __init__(self, net, transformer, init_prop, args, template_store, print_result=False):
        super().__init__(net, transformer, init_prop, args, template_store, print_result)
        # work set
        self.W = []
        self.L = len(self.cur_specs)
        self.node_visited = 1 # MCTS start with bab with subproblem, defaultly count for one original problem
        # hyperparameter
        self.sigma = 0.5 #! here
        self.c = 0.2 #! here
        self.reward = dict()
    def evaluate_leaf_set(self, leaf_set):
        if len(leaf_set) ==0:
            return True
        all_negative_infinity = all(reward == float('-inf') for reward in leaf_set.values())
        any_positive_infinity = any(reward == float('inf') for reward in leaf_set.values())

        if all_negative_infinity:
            self.global_status = Status.VERIFIED
            return False
        elif any_positive_infinity:
            self.global_status = Status.ADV_EXAMPLE
            return False
        else:
            return True
        
    def continue_search(self):
        return self.evaluate_leaf_set(self.reward) and self.global_status == Status.UNKNOWN and (not self.is_timeout())
    
    def helper(self, node, tree_set):
        node_name = str(node).split()[-1]
        if node_name not in tree_set:
            tree_set[node_name] = list()  
        if len(node.children) > 1: 
            for i in node.get_children():
                child_node = str(i).split()[-1]
                tree_set[child_node] = list()  
                tree_set[node_name].append(child_node)
                self.helper(i, tree_set)
        return 

    def verify_node(self, transformer, prop):
        """
        It is called from bnb_relu_complete. Attempts to verify (ilb, iub), there are three possible outcomes that
        are indicated by the status: 1) verified 2) adversarial example is found 3) Unknown
        """
        lb, is_feasible, adv_ex = transformer.compute_lb(complete=True)
        status = self.get_status(adv_ex, is_feasible, lb)
        if lb is not None:
            lb = float(lb)
        self.node_visited += 1
        return status, lb
    
    def assessMCTS(self,spec):
        
        if len(spec.children) == 2:
            spec_s = self.select_UCB1_child(spec, [spec.get_children()[0], spec.get_children()[1]])
            self.assessMCTS(spec_s)
        else: 
            if spec.status == Status.VERIFIED:
                self.global_status = spec.status
                return
            split_score = self.set_split_score(self.init_prop, self.cur_specs, inp_template=self.inp_template)
            assert(len(spec.children)==0)
            spec_a, spec_b = verifier_util.split_spec(spec=spec, split_type=self.split, split_score=split_score,
                                                                        inp_template=self.inp_template, args=self.args,
                                                                        net=self.net, transformer=self.transformer)
            for i in [spec_a, spec_b]:
                self.update_transformer(i.input_spec, relu_spec=i.relu_spec)
                status, lb = self.verify_node(self.transformer, i.input_spec)
                if status == Status.ADV_EXAMPLE:
                    self.global_status = status
                    self.store_final_tree()
                    print(f"Finished Verified Specicications, visited nodes: {self.node_visited}, find a counterexample")
                    return False
                i.update_status(status, lb)
                self.update_cur_lb(lb)
                print("\nMCTS SpecNode \t Depth: ", spec.depth, "LB: ", lb, "status: ", status)
                reward = self.compute_reward(status, lb, spec)
                self.reward[i] = reward
        self.reward[spec] = max(self.reward[spec.get_children()[0]], self.reward[spec.get_children()[1]])
        spec.mctsVisited += 2
    
    def compute_reward(self, status, lb, spec):
        if status == Status.ADV_EXAMPLE:
            return float("inf")
        elif status == Status.VERIFIED:
            return float("-inf")
        else:
            rootnode = spec.get_root()
            mini_lb_node = verifier_util.get_mini_lb(rootnode)
            # mini_lb = 0 ?
            assert(mini_lb_node.lb != 0)
            # We use total relus or unstable relus
            num_relus = len(self.get_unstable_relus())
            #TODO: get_verified_relus should traverse back to the parent root node? 
            return self.sigma*spec.depth/num_relus + (1-self.sigma)*lb/mini_lb_node.lb
    
    def select_UCB1_child(self, spec, spec_list):
        assert(len(spec_list) == 2)
        child1, child2 = spec_list[0], spec_list[1]
        value1 = self.reward[child1] + self.c*math.sqrt(2*math.log(spec.mctsVisited)/child1.mctsVisited)
        value2 = self.reward[child2] + self.c*math.sqrt(2*math.log(spec.mctsVisited)/child2.mctsVisited)
        return child1 if value1 > value2 else child2
    
    def run(self):
        """
        It is the public method called from the analyzer. @param split is a string that chooses the mode for relu
        or input splitting.
        """
        if self.global_status != Status.UNKNOWN:
            return
        
        assert len(self.cur_specs) == 1, "not the root node? in BNBMCTS run()"
        node = self.cur_specs[0]
        while self.continue_search():
            self.assessMCTS(node)
        print(f"BnB Balance Finished Verified Specicications, visited nodes: {self.node_visited}")
        self.check_verified_status()
        self.store_final_tree()
        return 

class Analyzer_MCTS_configure(Analyzer_MCTS):
    def __init__(self, args, net=None, template_store=None, lambda_config = 0.5, c_config=0.2):
        self.lambda_config = lambda_config
        self.c_config = c_config
        super().__init__(args, net=None, template_store=None)
    def analyze(self, prop):
        self.update_transformer(prop)
        tree_size = 1
        node_visited= 1
        
        # Check if classified correctly
        if nnverify.attack.check_adversarial(prop.input, self.net, prop):
            return Status.MISS_CLASSIFIED, tree_size, node_visited, None

        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if nnverify.attack.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size, node_visited, None

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            bnb_analyzer = BnB_MCTS_configure(self.net, self.transformer, prop, self.args, self.template_store, lambda_config=self.lambda_config, c_config=self.c_config)
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()

            status = bnb_analyzer.global_status
            tree_size = bnb_analyzer.tree_size
            node_visited = bnb_analyzer.node_visited
            lb = bnb_analyzer.cur_lb
        return status, tree_size, node_visited, lb
    
class BnB_MCTS_configure(BnB_MCTS):
    def __init__(self, net, transformer, init_prop, args, template_store, print_result=False, lambda_config=0.5, c_config=0.2):
        super().__init__(net, transformer, init_prop, args, template_store, print_result)
        self.sigma = lambda_config
        self.c = c_config

def summarize_results(file_name, image_index, eps, bab, mcts):
    with open(file_name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([image_index, eps,
                             "Baseline:", bab.time, bab.visited, bab.ver_output, bab.lb,
                             "MCTS:", mcts.time, mcts.visited, mcts.ver_output, mcts.lb] )

def summarize_result_mcts_hp(file_name, image_index, eps, lam_0_c_0, lam_0_c_02, lam_0_c_1, lam_05_c_0, lam_05_c_02, lam_05_c_1, lam_1_c_0, lam_1_c_02, lam_1_c_1):
    with open(file_name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([image_index, eps,
                            "lam_0_c_0:", lam_0_c_0.time, lam_0_c_0.visited, lam_0_c_0.ver_output, lam_0_c_0.lb,
                            "lam_0_c_02:", lam_0_c_02.time, lam_0_c_02.visited, lam_0_c_02.ver_output, lam_0_c_02.lb,
                            "lam_0_c_1:", lam_0_c_1.time, lam_0_c_1.visited, lam_0_c_1.ver_output, lam_0_c_1.lb,
                            "lam_05_c_0:", lam_05_c_0.time, lam_05_c_0.visited, lam_05_c_0.ver_output, lam_05_c_0.lb,
                            "lam_05_c_02:", lam_05_c_02.time, lam_05_c_02.visited, lam_05_c_02.ver_output, lam_05_c_02.lb,
                            "lam_05_c_1:", lam_05_c_1.time, lam_05_c_1.visited, lam_05_c_1.ver_output, lam_05_c_1.lb,
                            "lam_1_c_0:", lam_1_c_0.time, lam_1_c_0.visited, lam_1_c_0.ver_output, lam_1_c_0.lb,
                            "lam_1_c_02:", lam_1_c_02.time, lam_1_c_02.visited, lam_1_c_02.ver_output, lam_1_c_02.lb,
                            "lam_1_c_1:", lam_1_c_1.time, lam_1_c_1.visited, lam_1_c_1.ver_output, lam_1_c_1.lb,] )

def summarize_result_mcts_hp2(file_name, image_index, eps, lam_0_c_04, lam_0_c_06, lam_0_c_08, lam_05_c_04, lam_05_c_06, lam_05_c_08, lam_1_c_04, lam_1_c_06, lam_1_c_08):
    with open(file_name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([image_index, eps,
                            "lam_0_c_04:", lam_0_c_04.time, lam_0_c_04.visited, lam_0_c_04.ver_output, lam_0_c_04.lb,
                            "lam_0_c_06:", lam_0_c_06.time, lam_0_c_06.visited, lam_0_c_06.ver_output, lam_0_c_06.lb,
                            "lam_0_c_08:", lam_0_c_08.time, lam_0_c_08.visited, lam_0_c_08.ver_output, lam_0_c_08.lb,
                            "lam_05_c_04:", lam_05_c_04.time, lam_05_c_04.visited, lam_05_c_04.ver_output, lam_05_c_04.lb,
                            "lam_05_c_06:", lam_05_c_06.time, lam_05_c_06.visited, lam_05_c_06.ver_output, lam_05_c_06.lb,
                            "lam_05_c_08:", lam_05_c_08.time, lam_05_c_08.visited, lam_05_c_08.ver_output, lam_05_c_08.lb,
                            "lam_1_c_04:", lam_1_c_04.time, lam_1_c_04.visited, lam_1_c_04.ver_output, lam_1_c_04.lb,
                            "lam_1_c_06:", lam_1_c_06.time, lam_1_c_06.visited, lam_1_c_06.ver_output, lam_1_c_06.lb,
                            "lam_1_c_08:", lam_1_c_08.time, lam_1_c_08.visited, lam_1_c_08.ver_output, lam_1_c_08.lb,] )
            
def summarize_result(file_name, image_index, eps, mcts):
    with open(file_name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([image_index, eps,
                             "MCTS:", mcts.time, mcts.visited, mcts.ver_output, mcts.lb] )

def summarize_result_md(file_name, image_index, eps, lam_05_c_0, lam_05_c_02):
    with open(file_name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([image_index, eps,
                            "lam_05_c_0:", lam_05_c_0.time, lam_05_c_0.visited, lam_05_c_0.ver_output, lam_05_c_0.lb,
                            "lam_05_c_02:", lam_05_c_02.time, lam_05_c_02.visited, lam_05_c_02.ver_output, lam_05_c_02.lb] )

def testing_MNIST(filename, inputFile, option = "mnist01"):
    if option == "mnist01":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnist03":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_03, domain=Domain.LP, approx=ap.Prune(1),
                                 dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                 pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL2":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option =="mnistL4":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L4, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL6":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L6, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistvgg16":
        pt_args = pt.TransferArgs(net=config.MNIST_VGG_16, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyer_Baseline++++++++++++++\n")
        analyzer = AnalyzerBase(args)
        resbaseline = analyzer.run_analyzer(image_index, eps)
        print(f"+++++++++++++++++run Analyer_MCTS++++++++++++++\n")
        analyzer_MCTS = Analyzer_MCTS(args)
        resMCTS = analyzer_MCTS.run_analyzer(image_index,eps)
        
        summarize_results(filename,image_index, eps, resbaseline.results_list[0], resMCTS.results_list[0])
        coutlines += 1
    return 

def testing_MNIST_Mcts(filename, inputFile, option = "mnist01"):
    if option == "mnist01":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnist03":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_03, domain=Domain.LP, approx=ap.Prune(1),
                                 dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                 pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL2":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option =="mnistL4":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L4, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL6":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L6, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    
    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        print(f"+++++++++++++++++run Analyzer_MCTS++++++++++++++\n")
        analyzer_MCTS = Analyzer_MCTS(args)
        resMCTS = analyzer_MCTS.run_analyzer(image_index,eps)
        
        summarize_result(filename,image_index, eps, 
                          resMCTS.results_list[0]) 
        
        coutlines += 1
    return 

def testing_OVAL(filename, inputFile, mode="easy", option="cifar10ovalbase") -> None:
    MLmodel = mode
    if option == "cifar10ovalbase": #base model has mode = "easy"/ mode = "med"/ mode = "hard"
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.OVAL_CIFAR, split=Split.RELU_ESIP_SCORE, count=467, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "cifar10ovalwide": # model = "wide"
        # assert(mode=="wide")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.OVAL_CIFAR, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "wide"
    elif option == "cifar10ovaldeep": # mode = "deep"
        # assert(mode=="deep")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.OVAL_CIFAR, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "deep"
    args = pt_args.get_verification_arg()
    
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyer_Baseline++++++++++++++\n")
        analyzer = AnalyzerBase(args)
        resbaseline = analyzer.run_analyzer(image_index, eps, mode = MLmodel)
        # print(f"+++++++++++++++++run Analyer_MCTS++++++++++++++\n")
        # analyzer_MCTS = Analyzer_MCTS(args)
        # resMCTS = analyzer_MCTS.run_analyzer(image_index,eps, mode = MLmodel)
        
        summarize_results(filename,image_index, eps, 
                          resbaseline.results_list[0])
    return 

def testing_cifar(filename, inputFile, mode="easy", option="cifar10ovalbase") -> None:
    MLmodel = mode
    if option == "cifar10ovalbase": #base model has mode = "easy"/ mode = "med"/ mode = "hard"
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=467, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "cifar10ovalwide": # model = "wide"
        # assert(mode=="wide")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "wide"
    elif option == "cifar10ovaldeep": # mode = "deep"
        # assert(mode=="deep")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "deep"
    args = pt_args.get_verification_arg()
    
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyer_Baseline++++++++++++++\n")
        analyzer = AnalyzerBase(args)
        resbaseline = analyzer.run_analyzer(image_index, eps, mode = MLmodel)
        print(f"+++++++++++++++++run Analyer_MCTS++++++++++++++\n")
        analyzer_MCTS = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.2)
        resMCTS = analyzer_MCTS.run_analyzer(image_index, eps, mode = MLmodel)
        summarize_results(filename,image_index, eps,
                          resbaseline.results_list[0], 
                          resMCTS.results_list[0])
    return 

def testing_MNIST_mcts_hp(filename, inputFile, option = "mnist01") -> None:
    if option == "mnist01":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnist03":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_03, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL2":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option =="mnistL4":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L4, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL6":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L6, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)

    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 0 ++++++++++++++\n")
        analyzer0 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0)
        res0 = analyzer0.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 1 ++++++++++++++\n")
        analyzer1 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.2)
        res1 = analyzer1.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 2 ++++++++++++++\n")
        analyzer2 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 1)
        res2 = analyzer2.run_analyzer(image_index, eps)

        print(f"+++++++++++++++++run Analyzer_mcts_hp 3 ++++++++++++++\n")
        analyzer3 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0)
        res3 = analyzer3.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 4 ++++++++++++++\n")
        analyzer4 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.2)
        res4 = analyzer4.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 5 ++++++++++++++\n")
        analyzer5 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 1)
        res5 = analyzer5.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 6 ++++++++++++++\n")
        analyzer6 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0)
        res6 = analyzer6.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 7 ++++++++++++++\n")
        analyzer7 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.2)
        res7 = analyzer7.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 8 ++++++++++++++\n")
        analyzer8 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 1)
        res8 = analyzer8.run_analyzer(image_index, eps)


        summarize_result_mcts_hp(filename, image_index, eps, 
                                                        res0.results_list[0], 
                                                        res1.results_list[0], 
                                                        res2.results_list[0], 
                                                        res3.results_list[0], 
                                                        res4.results_list[0], 
                                                        res5.results_list[0],
                                                        res6.results_list[0], 
                                                        res7.results_list[0], 
                                                        res8.results_list[0],)
        
        coutlines += 1


def testing_MNIST_mcts_hp2(filename, inputFile, option = "mnist01") -> None:
    if option == "mnist01":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnist03":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_03, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL2":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option =="mnistL4":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L4, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL6":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L6, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)

    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 0 ++++++++++++++\n")
        analyzer0 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.4)
        res0 = analyzer0.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 1 ++++++++++++++\n")
        analyzer1 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.6)
        res1 = analyzer1.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 2 ++++++++++++++\n")
        analyzer2 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.8)
        res2 = analyzer2.run_analyzer(image_index, eps)

        print(f"+++++++++++++++++run Analyzer_mcts_hp 3 ++++++++++++++\n")
        analyzer3 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.4)
        res3 = analyzer3.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 4 ++++++++++++++\n")
        analyzer4 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.6)
        res4 = analyzer4.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 5 ++++++++++++++\n")
        analyzer5 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.8)
        res5 = analyzer5.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 6 ++++++++++++++\n")
        analyzer6 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.4)
        res6 = analyzer6.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 7 ++++++++++++++\n")
        analyzer7 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.6)
        res7 = analyzer7.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 8 ++++++++++++++\n")
        analyzer8 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.8)
        res8 = analyzer8.run_analyzer(image_index, eps)


        summarize_result_mcts_hp2(filename, image_index, eps, 
                                                        res0.results_list[0], 
                                                        res1.results_list[0], 
                                                        res2.results_list[0], 
                                                        res3.results_list[0], 
                                                        res4.results_list[0], 
                                                        res5.results_list[0],
                                                        res6.results_list[0], 
                                                        res7.results_list[0], 
                                                        res8.results_list[0],)
        
        coutlines += 1


def testing_CIFAR_mcts_hp(filename, inputFile, mode="easy", option="cifar10ovalbase") -> None:
    MLmodel = mode
    if option == "hp_base": #base model has mode = "easy"/ mode = "med"/ mode = "hard"
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=467, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "hp_wide": # model = "wide"
        # assert(mode=="wide")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "wide"
    elif option == "hp_deep": # mode = "deep"
        # assert(mode=="deep")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "deep"

    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 0 ++++++++++++++\n")
        analyzer0 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0)
        res0 = analyzer0.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 1 ++++++++++++++\n")
        analyzer1 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.2)
        res1 = analyzer1.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 2 ++++++++++++++\n")
        analyzer2 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 1)
        res2 = analyzer2.run_analyzer(image_index, eps, mode = MLmodel)

        print(f"+++++++++++++++++run Analyzer_mcts_hp 3 ++++++++++++++\n")
        analyzer3 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0)
        res3 = analyzer3.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 4 ++++++++++++++\n")
        analyzer4 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.2)
        res4 = analyzer4.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 5 ++++++++++++++\n")
        analyzer5 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 1)
        res5 = analyzer5.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 6 ++++++++++++++\n")
        analyzer6 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0)
        res6 = analyzer6.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 7 ++++++++++++++\n")
        analyzer7 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.2)
        res7 = analyzer7.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 8 ++++++++++++++\n")
        analyzer8 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 1)
        res8 = analyzer8.run_analyzer(image_index, eps, mode = MLmodel)


        summarize_result_mcts_hp(filename, image_index, eps, 
                                                        res0.results_list[0], 
                                                        res1.results_list[0], 
                                                        res2.results_list[0], 
                                                        res3.results_list[0], 
                                                        res4.results_list[0], 
                                                        res5.results_list[0],
                                                        res6.results_list[0], 
                                                        res7.results_list[0], 
                                                        res8.results_list[0],)
        
        coutlines += 1


def testing_CIFAR_mcts_hp2(filename, inputFile, mode="easy", option="cifar10ovalbase") -> None:
    MLmodel = mode
    if option == "hp2_base": #base model has mode = "easy"/ mode = "med"/ mode = "hard"
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=467, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "hp2_wide": # model = "wide"
        # assert(mode=="wide")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "wide"
    elif option == "hp2_deep": # mode = "deep"
        # assert(mode=="deep")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "deep"

    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 0 ++++++++++++++\n")
        analyzer0 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.4)
        res0 = analyzer0.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 1 ++++++++++++++\n")
        analyzer1 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.6)
        res1 = analyzer1.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 2 ++++++++++++++\n")
        analyzer2 = Analyzer_MCTS_configure(args, lambda_config = 0, c_config = 0.8)
        res2 = analyzer2.run_analyzer(image_index, eps, mode = MLmodel)

        print(f"+++++++++++++++++run Analyzer_mcts_hp 3 ++++++++++++++\n")
        analyzer3 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.4)
        res3 = analyzer3.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 4 ++++++++++++++\n")
        analyzer4 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.6)
        res4 = analyzer4.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 5 ++++++++++++++\n")
        analyzer5 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.8)
        res5 = analyzer5.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 6 ++++++++++++++\n")
        analyzer6 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.4)
        res6 = analyzer6.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 7 ++++++++++++++\n")
        analyzer7 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.6)
        res7 = analyzer7.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_mcts_hp 8 ++++++++++++++\n")
        analyzer8 = Analyzer_MCTS_configure(args, lambda_config = 1, c_config = 0.8)
        res8 = analyzer8.run_analyzer(image_index, eps, mode = MLmodel)


        summarize_result_mcts_hp2(filename, image_index, eps, 
                                                        res0.results_list[0], 
                                                        res1.results_list[0], 
                                                        res2.results_list[0], 
                                                        res3.results_list[0], 
                                                        res4.results_list[0], 
                                                        res5.results_list[0],
                                                        res6.results_list[0], 
                                                        res7.results_list[0], 
                                                        res8.results_list[0],)
        
        coutlines += 1

def testing_MNIST_md(filename, inputFile, option = "mnist01") -> None:
    if option == "mnist01":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnist03":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_03, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "md_mnistl2":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option =="md_mnistl4":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L4, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "mnistL6":
        pt_args = pt.TransferArgs(net=config.MNIST_FFN_L6, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.MNIST, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)

    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]

        print(f"+++++++++++++++++run Analyzer_l_05_c_0 ++++++++++++++\n")
        analyzer_l_05_c_0 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0)
        res_l_05_c_0 = analyzer_l_05_c_0.run_analyzer(image_index, eps)
        
        print(f"+++++++++++++++++run Analyzer_l_05_c_02 ++++++++++++++\n")
        analyzer_l_05_c_02 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.2)
        res_l_05_c_02 = analyzer_l_05_c_02.run_analyzer(image_index, eps)


        summarize_result_md(filename, image_index, eps,  
                                    res_l_05_c_0.results_list[0], 
                                    res_l_05_c_02.results_list[0])
        
        coutlines += 1

def testing_CIFAR_md(filename, inputFile, mode="easy", option="cifar10ovalbase") -> None:
    MLmodel = mode
    if option == "md_base": #base model has mode = "easy"/ mode = "med"/ mode = "hard"
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=467, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
    elif option == "md_wide": # model = "wide"
        # assert(mode=="wide")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "wide"
    elif option == "md_deep": # mode = "deep"
        # assert(mode=="deep")
        pt_args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP, approx=ap.Prune(1),
                                dataset=Dataset.CIFAR10, split=Split.RELU_ESIP_SCORE, count=100, 
                                pt_method=ProofTransferMethod.REUSE, timeout=1000)
        MLmodel = "deep"

    args = pt_args.get_verification_arg()
    lines = list()
    with open(inputFile, "r") as f:
        lines = f.readlines()
    coutlines = 0

    for line in lines:
        line_data = eval(line.strip())
        line_data = [int(line_data[0]), float(line_data[1])]
        print(f"============current processing : {line_data},  in the line {coutlines}==============")
        
        image_index = line_data[0]
        eps = line_data[1]

        print(f"+++++++++++++++++run Analyzer_l_05_c_0 ++++++++++++++\n")
        analyzer_l_05_c_0 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0)
        res_l_05_c_0 = analyzer_l_05_c_0.run_analyzer(image_index, eps, mode = MLmodel)
        
        print(f"+++++++++++++++++run Analyzer_l_05_c_02 ++++++++++++++\n")
        analyzer_l_05_c_02 = Analyzer_MCTS_configure(args, lambda_config = 0.5, c_config = 0.2)
        res_l_05_c_02 = analyzer_l_05_c_02.run_analyzer(image_index, eps, mode = MLmodel)


        summarize_result_md(filename, image_index, eps, 
                                                        res_l_05_c_0.results_list[0], 
                                                        res_l_05_c_02.results_list[0])
        
        coutlines += 1


if __name__ == '__main__':
    program_option = sys.argv[1]
    if len(sys.argv) > 2:
        file_number = sys.argv[2]
    
    if program_option == "mnist01" or program_option == "mnist03" or program_option == "mnistL2" or program_option == "mnistL4" or program_option == "mnistL6" or program_option=="mnistvgg16":
        if len(file_number) == 2: #01-17
            file_number = int(file_number)
            testing_MNIST(f"Result_{program_option}_{file_number}.csv", 
                      f"./data/treeSpecification/{program_option}_{file_number}.txt", 
                      option=program_option)
        elif len(file_number) ==3: #001 017
            file_number = int(file_number)
            testing_MNIST_Mcts(f"Result_{program_option}_{file_number}.csv", 
                      f"./data/treeSpecification/{program_option}_{file_number}.txt", 
                      option=program_option)
            
    elif program_option == "cifar10ovalbase" or program_option == "cifar10ovaldeep" or program_option == "cifar10ovalwide":
    #     if len(file_number) == 1:
    #         testing_OVAL(f"Result_{program_option}_{file_number}_oval.csv", 
    #                     f"./data/treeSpecification/{program_option}_{file_number}.txt", 
    #                     mode="easy", option=program_option)
        
        if len(file_number) ==2: #01 02
            file_number = int(file_number)
            testing_cifar(f"Result_{program_option}_{file_number}_cifar.csv", 
                        f"./data/treeSpecification/{program_option}_{file_number}.txt", 
                        mode="easy", option=program_option)
    elif program_option == "hp_mnistl2":
        testing_MNIST_mcts_hp(f"Result_Mcts_{program_option}.csv", 
                              f"./data/treeSpecification/{program_option}.txt", 
                              option='mnistL2')
    elif program_option == "hp_mnistl4":
        testing_MNIST_mcts_hp(f"Result_Mcts_{program_option}.csv", 
                              f"./data/treeSpecification/{program_option}.txt", 
                              option='mnistL4')
    elif program_option == "hp2_mnistl2":
        testing_MNIST_mcts_hp2(f"Result_Mcts_{program_option}.csv", 
                              f"./data/treeSpecification/{program_option}.txt", 
                              option='mnistL2')
    elif program_option == "hp2_mnistl4":
        testing_MNIST_mcts_hp2(f"Result_Mcts_{program_option}.csv", 
                              f"./data/treeSpecification/{program_option}.txt", 
                              option='mnistL4')
    elif program_option == "hp_base" or program_option == "hp_deep" or program_option == "hp_wide":
        testing_CIFAR_mcts_hp(f"Result_Mcts_{program_option}.csv", 
                              f"./data/treeSpecification/{program_option}.txt", 
                              mode='easy',
                              option=program_option)
    elif program_option == "hp2_base" or program_option == "hp2_deep" or program_option == "hp2_wide":
        testing_CIFAR_mcts_hp2(f"Result_Mcts_{program_option}.csv", 
                              f"./data/treeSpecification/{program_option}.txt", 
                              mode='easy',
                              option=program_option)
    elif program_option == "md_mnistl2" or program_option == "md_mnistl4":
        testing_MNIST_md(f"Result_{program_option}_{file_number}.csv", 
                              f"./data/treeSpecification/mdMnist/{program_option}_{file_number}.txt", 
                              option=program_option)
    elif program_option == "md_base" or program_option == "md_deep" or program_option == "md_wide":
        testing_CIFAR_md(f"Result_{program_option}_{file_number}.csv", 
                              f"./data/treeSpecification/mdCifar/{program_option}_{file_number}.txt", 
                              mode='easy',
                              option=program_option)
