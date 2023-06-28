import os
from collections import Counter
BASE_DIRECTORY = "/work/mbirlikc/data/"
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, r'Processed Interviews')

def check_status(categories):
    counter = Counter()
    result = {}
    
    for input_category in categories:
        output_directory = os.path.join(OUTPUT_DIRECTORY, input_category)
        
        # Get a list of all directories in the output directory
        files = os.listdir(output_directory)
        
        # Filter out directories and obtain the full file paths in source directory
        file_directories = [file_name for file_name in files if os.path.isdir(os.path.join(output_directory, file_name))]
        
        # The number of jobs is the number of directories
        counter[input_category + " Jobs"] = len(file_directories)
        
        # The number of completed is the number of directories 
        # that have a landmark.pkl file
        completed_jobs = [file_name for file_name in file_directories if os.path.exists(os.path.join(output_directory, file_name, "landmark.pkl"))]
        counter[input_category + " Completed"] = len(completed_jobs)
        
        # Format it as the percentage of completed jobs versus total jobs
        # as a string with 2 decimal places
        result[input_category] = "{:.2f}%".format(100 * counter[input_category + " Completed"] / counter[input_category + " Jobs"])
    
    return result, counter

# Write code that runs if .py file is run as a script
if __name__ == "__main__":
    # categories = ["Two-Player"]
    categories = ["One-Player", "On-Court", "Need-Classification"]
    categories = ["One-Player", "On-Court"]

    # get the list of video_files
    percentage, job_count = check_status(categories)
    
    print("Job status (in percentage):")
    print(percentage)
    
    print("Detailed breakdown (in file count):")
    print(job_count)
    