"""
This is a helper to run the searches in batch and automatically collect results in csv file.
I chose to run 4 search methods 6 times for each of 3 cargo problems.

Note, run_search was written by Udacity without thinking of capturing output, mixing run objects with print() in stdout
thus some mess is here as well.
"""

from pathlib import Path
import re
from contextlib import redirect_stdout
from multiprocessing import Pool

from run_search import main as run_search_main

do_search = True # may wish to parse old log files istead
runs = 6 # better be less than num CPUS
problems = ['1', '2', '3', '4' ] # cargo problems
searches = ['9', '10', '3', '7', '1', '5'] 
# 9, 10 = heuristic search using A* with the "ignore preconditions" and "level-sum" heuristics,
# 1-7 uninformed search methods: bfs uniform_cost_search dfs greedy_best_first_graph_search_h_1 
log_dir = Path("./log")
results_csv = log_dir / "search_result.csv"
batches = [{"id":n, "log_file": log_dir/ f"search-{n}.log"} for n in range(runs)]

def run_batch(log_file: Path):
    """ executes search metods that dumps output in text file
    """
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            print(f"run_search_main({problems}, {searches})")
            #TODO terminate by timeout
            run_search_main(problems, searches)

def csv_header():
    with open(results_csv, "w+") as result_f:
        result_f.write("run, problem, method, Expansions, GoalTests, NewNodes, plan_size, time_elapsed,\n")

def csv_results(id: int, output: Path):
    """ parse output of an run with few search methods
    log results in format:
    run, problem, method, Expansions, GoalTests, NewNodes, plan_size, time_elapsed,
    """
    problem_re = re.compile("^Solving Air Cargo Problem (\d+) using (\w+)(?: with (\w+))*.")
    search_re = re.compile("^Plan length: (\d+)  Time elapsed in seconds: (\d+\.?\d*)*.")
    # parsing 3 output lines into one csv live
    part = 1
    with open(results_csv, "a") as result_f:
        with open(output) as output_f:
            for line in output_f:
                if part ==1:
                    m = problem_re.match(line)
                    if m: # "Solving Air Cargo Problem 1 using depth_first_graph_search..."
                        problem = m.group(1)
                        method = m.group(2)
                        if m.group(3):
                            method += '_'+m.group(3)
                        result_f.write(f"{id},p{problem},{method},")
                        part = 2
                if part ==2:
                    if line.startswith("Expansions   Goal Tests   New Nodes"):
                        # parse next line  " 43 56 180 " 
                        vals = output_f.readline().split()
                        result_f.write(f"{vals[0]},{vals[1]},{vals[2]},")
                        part = 3
                if part ==3:
                    m = search_re.match(line)
                    if m: # "Plan length: 12  Time elapsed in seconds: 0.024"
                        plan_size = m.group(1)
                        time_elapsed = m.group(2)
                        result_f.write(f"{plan_size},{time_elapsed}")
                        result_f.write("\n")
                        part = 1
        # terminate uncomplete records
        if part !=1:
            result_f.write("\n")
        
                    
if __name__=="__main__":
    if not log_dir.exists():
        print(f'ERR: log dir does not exist {log_dir}') 
    else:
        if do_search:
            # batches are exec in parallel
            with Pool(runs) as p:
                p.map( run_batch, [b["log_file"] for b in batches] )
        # collect results
        csv_header()
        for b in batches:
            log_file = b["log_file"]
            id = b["id"]
            csv_results(id, log_file)
        print('OK: all batches done, normal exit.')

