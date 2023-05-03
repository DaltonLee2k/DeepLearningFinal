import csv
import os




# Open the input and output files
with open('qbInput.csv', 'r') as input_file, open('output.csv', 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Loop over each row in the input file
    for row in reader:
        # Append a comma and a 0 to the end of the row
        row.append(',')
        row.append('0')
        # Write the modified row to the output file
        writer.writerow(row)
