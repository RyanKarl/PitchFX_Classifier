import pandas
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

filename = args.file
df = pandas.read_csv(filename)

for i in range(522,678):
    print(df.columns[i])
    df.drop(df.columns[i], axis=1)

print(df)

path = './clean' + filename

export_csv = df.to_csv(path , index = None, header=True)


