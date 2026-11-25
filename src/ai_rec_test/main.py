from .services.parse_data import parse_table
from .services import P1, P2, P3, P4, P5


def main():
    df_problem = parse_table()
    P1.solve(df_problem)
    P2.solve(df_problem)
    P3.solve(df_problem)
    P4.solve(df_problem)
    P5.solve(df_problem)



if __name__=="__main__":
    main()