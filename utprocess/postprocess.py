import utprocess
import matplotlib.pyplot as plt

filename = "test_output_new.json"

peaks = utprocess.PeaksActive.from_json(fname=filename)

peaks.party_time()
peaks.write_to_csv_utinvert("test.csv")