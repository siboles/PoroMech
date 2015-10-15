import PoroMech
import pickle

p = PoroMech.Data("CyclicComp1.tdms")

g = p.time.keys()[0]
c = p.time[g].keys()

fid = open("data.pkl", "wb")
pickle.dump({'LVDT': [p.time[g]['LVDT'], p.data[g]['LVDT']],
             'Load Cell': [p.time[g]['Load Cell'], p.data[g]['Load Cell']]},
            fid, 2)
fid.close()
