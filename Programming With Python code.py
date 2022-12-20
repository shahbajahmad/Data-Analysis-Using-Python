import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as Plott
from sqlalchemy import create_engine


class regression:
    def init(self):
        pass

    def MatchIdeal(self, f_training, f_ideal):

        if isinstance(f_training, pd.DataFrame) and isinstance(f_ideal, pd.DataFrame):
            Column_Ideal = len(f_ideal.columns)
            Row_Training = f_training.index[-1] + 1
            Column_Training = len(f_training.columns)
            ListOfIndex = list()
            LS = list()
            for col in range(1, Column_Training):
                LS1 = list()
                for col2 in range(1, Column_Ideal):
                    SMSE = 0
                    for i in range(Row_Training):
                        zvalue1 = f_training.iloc[i, col]
                        zvalue2 = f_ideal.iloc[i, col2]
                        SMSE = SMSE + ((zvalue1 - zvalue2) ** 2)
                    LS1.append(SMSE / Row_Training)
                LeastMin = min(LS1)
                Idx = LS1.index(LeastMin)
                ListOfIndex.append(Idx + 1)
                LS.append(LeastMin)

            PPframe = pd.DataFrame(list(zip(ListOfIndex, LS)), columns=[
                                   "Index", "LS_value"])

            return PPframe
        else:
            raise TypeError("Data frame is wrong type!")

    def MatchingRows(self, f_test):

        if isinstance(f_test, pd.DataFrame):
            Row_Testing = f_test.index[-1] + 1
            Column_Testing = len(f_test.columns)
            IndexOfIdeal = list()
            DIV_ = list()
            for row in range(Row_Testing):
                MSE_l = list()
                for row1 in range(2, Column_Testing):
                    zvalue1 = f_test.iloc[row, 1]
                    zvalue2 = f_test.iloc[row, row1]
                    SMSE = ((zvalue2 - zvalue1) ** 2)
                    MSE_l.append(SMSE)
                LeastMin = min(MSE_l)
                if LeastMin < (np.sqrt(2))*0.8:
                    DIV_.append(LeastMin)
                    index = MSE_l.index(LeastMin)
                    IndexOfIdeal.append(index)
                else:
                    DIV_.append(LeastMin)
                    IndexOfIdeal.append("Miss")
            f_test["DIV_"] = DIV_
            f_test["Ideal_Index"] = IndexOfIdeal

            return f_test

        else:
            raise TypeError("Data frame is wrong type!")

    def PlottingGraphs(self, functionX, ParameterX, functionY1, ParameterY1, functionY2, ParameterY2, Plot_show_=True):

        funx = functionX.iloc[:, ParameterX]
        funy1 = functionY1.iloc[:, ParameterY1]
        funy2 = functionY2.iloc[:, ParameterY2]

        Plott.plot(funx, funy1, c="r", label="Train_Function")
        Plott.plot(funx, funy2, c="b", label="Ideal_Functiontion")
        Plott.xlabel("x")
        Plott.ylabel("y")
        Plott.legend(loc=3)

        if Plot_show_ is True:
            Plott.show()
            Plott.clf()
        elif Plot_show_ is False:
            pass
        else:
            pass


class DBCreation_SQLITE(regression):

    def CreatTableDB(self, df, DBNM, TNM):

        try:
            engine = create_engine(f"sqlite:///{DBNM}.db", echo=True)
            sqlite_connection = engine.connect()
            for i in range(len(df)):
                dfz = df[i]
                dfz.to_sql(TNM[i], sqlite_connection, if_exists="fail")
            sqlite_connection.close()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_type, exc_value, exc_traceback)


tr = pd.read_csv("train.csv")
Idl = pd.read_csv("ideal.csv")
te = pd.read_csv("test.csv")


df = regression().MatchIdeal(tr, Idl)
print(df)

G = regression()
for it in range(1, len(tr.columns)):
    G.PlottingGraphs(tr, 0, tr, it, Idl, df.iloc[it-1, 0], False)

te = te.sort_values(by=["x"], ascending=True)
te = te.reset_index()
te = te.drop(columns=["index"])

Idls = list()
for it in range(0, 4):
    Idls.append(Idl[["x", f"y{str(df.iloc[it, 0])}"]])

for it in Idls:
    te = te.merge(it, on="x", how="left")

te = regression().MatchingRows(te)

for it in range(0, 4):
    te["Ideal_Index"] = te["Ideal_Index"].replace(
        [it], str(f"y{df.iloc[it, 0]}"))


te_st_ = te
te_st_["Idl y value"] = ""
for it in range(0, 100):
    kt = te_st_.iloc[it, 7]
    if kt == "y14":
        te_st_.iloc[it, 8] = te_st_.iloc[it, 2]
    elif kt == "y29":
        te_st_.iloc[it, 8] = te_st_.iloc[it, 3]
    elif kt == "y3":
        te_st_.iloc[it, 8] = te_st_.iloc[it, 4]
    elif kt == "y38":
        te_st_.iloc[it, 8] = te_st_.iloc[it, 5]
    elif kt == "Miss":
        te_st_.iloc[it, 8] = te_st_.iloc[it, 1]

tr = tr.rename(columns={"y1": "Y1 (Training_Func)", "y2": "Y2 (Training_Func)",
                              "y3": "Y3 (Training_Func)", "y4": "Y4 (Training_Func)"})


for col in Idl.columns:
    if len(col) > 1:
        Idl = Idl.rename(columns={col: f"{col} (Ideal_Function)"})

te = te.rename(columns={"x": "X (Test_Function)",
                        "y": "Y (Test_Function)",
                        "DIV_": "Delta Y (Test_Function)",
                        "Idl index": "No. of Ideal_Function"})
# Load data to sqlite
dbs = DBCreation_SQLITE()
dfms = [tr, Idl, te]
TNMs = ["tr_table", "Idl_table", "te_table"]
dbs.CreatTableDB(dfms, "functiond_database", TNMs)

# Visualization
# Training_functions
Plott.clf()
x_data = tr.iloc[:, 0]
for it in range(1, len(tr.columns)):
    Plott.plot(x_data, tr.iloc[:, it], c="g", label=f"Training_function y{it}")
    Plott.legend(loc=3)
    Plott.show()
    Plott.clf()

# Ideal_Functiontion
for it in range(1, 5):
    bb = Idl.keys()
    Plott.plot(x_data, Idl.iloc[:, it], c="#FF4500", label=f"{bb[it]}")
    Plott.legend(loc=3)
    Plott.show()
    Plott.clf()


Plott.clf()
Plott.scatter(te.iloc[:, 0], te.iloc[:, 1])
Plott.show()

Plott.clf()
# construct lists to visualise te scat dataframe
x1_data = list()
x2_data = list()
x3_data = list()
x4_data = list()
xm_data = list()
y1_data = list()
y2_data = list()
y3_data = list()
y4_data = list()
ym_data = list()

# add the x and y values to the lists that are above it.
for it in range(0, 100):
    kt = te_st_.iloc[it, 7]
    if kt == "y14":
        x1_data.append(te_st_.iloc[it, 0])
        y1_data.append(te_st_.iloc[it, 8])
    elif kt == "y29":
        x2_data.append(te_st_.iloc[it, 0])
        y2_data.append(te_st_.iloc[it, 8])
    elif kt == "y3":
        x3_data.append(te_st_.iloc[it, 0])
        y3_data.append(te_st_.iloc[it, 8])
    elif kt == "y38":
        x4_data.append(te_st_.iloc[it, 0])
        y4_data.append(te_st_.iloc[it, 8])
    elif kt == "Miss":
        xm_data.append(te_st_.iloc[it, 0])
        ym_data.append(te_st_.iloc[it, 8])

# Ideal_Functiontions and y-values should both be plotted on the same scatter plot.
Plott.scatter(x1_data, y1_data, marker="o", label="test - y14", color="r")
Plott.scatter(x2_data, y2_data, marker="s", label="test - y29", color="b")
Plott.scatter(x3_data, y3_data, marker="^", label="test - y3", color="g")
Plott.scatter(x4_data, y4_data, marker="d",
              label="test - y38", color="#FFD700")
Plott.scatter(xm_data, ym_data, marker="x",
              label="test - Miss", color="#000000")
Plott.plot(Idl.iloc[:, 0], Idl.iloc[:, 14],
           label="Ideal - Y14", color="#FA8072")
Plott.plot(Idl.iloc[:, 0], Idl.iloc[:, 29],
           label="Ideal - Y29", color="#1E90FF")
Plott.plot(Idl.iloc[:, 0], Idl.iloc[:, 3], label="Ideal - Y3", color="#7CFC00")
Plott.plot(Idl.iloc[:, 0], Idl.iloc[:, 38],
           label="Ideal - Y38", color="#FFA500")
Plott.xlabel("x")
Plott.xlabel("y")
Plott.legend()
Plott.show()
