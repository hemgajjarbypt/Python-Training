import pandas as pd

df = pd.read_csv("students.csv")

print("List of Students:", len(df))
print(df.head(50))


older_students = df[df['Age'] > 18]
print("List of Students older than 18 Year:", len(older_students))
print(older_students)