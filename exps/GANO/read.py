import re

def extract_unique_graph_filenames(log_path):
    graph_files = set()  # use a set for uniqueness

    with open(log_path, 'r') as file:
        for line in file:
            # Split the line by both '/' and ':'
            parts = re.split(r'[/:]', line)
            for part in parts:
                if part.startswith('graph') and part.endswith('.pt'):
                    graph_files.add(part)
                    break  # stop after the first match in a line

    return sorted(graph_files)  # return as a sorted list

# Example usage:
log_file = './driver_plus_GANO.out'  # replace with your actual log file
bad_graph_ids = extract_unique_graph_filenames(log_file)
print("Problematic graph IDs:")
print(bad_graph_ids)
print('number of graphs:', len(bad_graph_ids))