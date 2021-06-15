from pymoo.algorithms.PushPullSearch import PushSearch,PullSearch
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.termination.pps_termination import PushPullSearchTermination
from pymoo.model.callback import Callback

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def notify(self, algorithm):
        print(algorithm.pop.get("X").shape)
        self.data.append(algorithm.pop.get("X"))

def minimize_PPS(problem,pop_size = 300,seed = 1,verbose = False,n_gen_pull = 100):
    problem = problem

    algorithm = PushSearch(pop_size=pop_size)

    termination = PushPullSearchTermination()

    res = minimize(problem,
               algorithm,
               termination,
               seed=seed,
               callback=MyCallback(),
               save_history=False,
               verbose=verbose)

    val = res.algorithm.callback.data

    F = problem.evaluate(val[-1],return_values_of = ['F'])

    plot1 = Scatter()
    plot1.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot1.add(F, color="red")
    
    algorithm2 = PullSearch(pop_size = pop_size,
                           init_pop = val[-1] )

    res2  = minimize(problem,
                    algorithm2,
                    ('n_gen',200),
                    seed = seed,
                    verbose = verbose)

    plot2 = Scatter()
    plot2.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot2.add(res2.F, color="red")

    return res,plot1,res2,plot2,val
    
