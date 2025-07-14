import argparse
import pandas as pd
from topsisx.topsis import topsis
from topsisx.ahp import ahp
from topsisx.vikor import vikor
from topsisx.entropy import entropy_weights
from topsisx.reports import generate_report

def main():
    parser = argparse.ArgumentParser(description="TOPSISX CLI Tool")
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--method", "-m", choices=["topsis", "ahp", "vikor"], default="topsis")
    parser.add_argument("--weights", "-w", type=str, help="Comma-separated weights")
    parser.add_argument("--impacts", "-i", type=str, help="Comma-separated impacts (+/-)")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    if args.method == "topsis":
        weights = [float(w) for w in args.weights.split(",")] if args.weights else entropy_weights(df.iloc[:, 1:])
        impacts = [i.strip() for i in args.impacts.split(",")]
        result = topsis(df.iloc[:, 1:], weights, impacts)
    elif args.method == "vikor":
        weights = [float(w) for w in args.weights.split(",")] if args.weights else entropy_weights(df.iloc[:, 1:])
        impacts = [i.strip() for i in args.impacts.split(",")]
        result = vikor(df.iloc[:, 1:], weights, impacts)
    elif args.method == "ahp":
        pairwise = df.iloc[:, 1:]
        result = ahp(pairwise)

    print(result)

    if args.report:
        generate_report(result)

if __name__ == "__main__":
    main()
