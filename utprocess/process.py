import utprocess
import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level=logging.DEBUG)


folder = "test/data/vuws/"
filegroup = [[f"{x}.dat" for x in range(1, 6)],
             [f"{x}.dat" for x in range(6, 11)],
             [f"{x}.dat" for x in range(11, 16)],
             [f"{x}.dat" for x in range(16, 26)],
             [f"{x}.dat" for x in range(26, 36)],
             [f"{x}.dat" for x in range(36, 46)],
             ]

settings_file = "test/test_fksettings.json"

for gnum, group in enumerate(filegroup):
    for fnum, fname in enumerate(group):
        group[fnum] = folder+fname
    array = utprocess.Array1D.from_seg2s(group)
    # array.plot_array()
    # array.plot_waterfall()
    fk = utprocess.WavefieldTransform1D(array=array,
                                        settings_file=settings_file)
    # array.plot_waterfall()
    fk.plot_spec()
    fk.save_peaks(fname="test_output_new",
                  identifier=array.source.position["x"],
                  append=False if gnum == 0 else True)
plt.show()
