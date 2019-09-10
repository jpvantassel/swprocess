import utprocess
import matplotlib.pyplot as plt
import os

folder = "test/data/vuws/"
filegroup = [["1.dat", "2.dat", "3.dat", "4.dat", "5.dat"],
             ["6.dat", "7.dat", "8.dat", "9.dat", "10.dat"],
             ["11.dat", "12.dat", "13.dat", "14.dat", "15.dat"]
             ]
settings_file = "test/test_fksettings.json"

os.remove("test_output_new.json")

for group in filegroup:
    for fnum, fname in enumerate(group):
        group[fnum] = folder+fname
    array = utprocess.Array1D.from_seg2s(group)

    fk = utprocess.WavefieldTransform1D(array=array,
                                        settings_file=settings_file)
    fk.disp_power.plot_spec(plot_limit=[5, 100, 0, 500])
    fk.disp_power.save_peaks(fname="test_output_new",
                             source_location=array.source.x)
plt.show()
