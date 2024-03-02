import os, sys
import json
import pandas as pd


def json_data_loader(fp):

    with open(fp) as f:
        # list of dict objects
        data = json.load(f)

    print("loaded ", len(data), "conversations.")

    return data


d = json_data_loader("jira-conversations2.json")

# summarize all questions with in each conversation and also the answers.

# collect data from websites using webscraping
