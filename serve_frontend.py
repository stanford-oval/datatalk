from flask import Flask, render_template, send_from_directory, request, jsonify, Response, stream_with_context
import os
import sys
import jinja2
import re
import shutil
import tempfile
import time
import io
import threading
import queue
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.append(backend_dir)
sys.path.append(os.path.join(backend_dir, 'ingestion'))

# Now import the ingestion module
from backend.ingestion.ingestion import process_all_csvs

# Create Flask app with custom template loader
app = Flask(__name__)
app.template_folder = 'new_frontend/templates'
app.static_folder = 'new_frontend'

# Set up a custom Jinja2 loader to handle templates/ prefix
template_loader = jinja2.ChoiceLoader([
    app.jinja_loader,
    jinja2.FileSystemLoader('new_frontend'),
])
app.jinja_loader = template_loader

@app.route('/')
def index():
    return render_template('upload.jinja2')

@app.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory('new_frontend/js', path)

@app.route('/css/<path:path>')
def serve_css(path):
    return send_from_directory('new_frontend/css', path)

@app.route('/img/<path:path>')
def serve_img(path):
    return send_from_directory('new_frontend/img', path)

@app.route('/favicon.png')
def favicon():
    return send_from_directory('new_frontend', 'favicon.png')

# File upload configuration
MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10  # 10GB
ALLOWED_EXTENSIONS = {".csv"}

def is_allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Parse the form data to get the database name and files
        database_name = request.form.get("database")
        
        if not database_name:
            return jsonify({"success": False, "error": "Database name is required"})
        
        # Sanitize database name to avoid path traversal attacks
        database_name = generate_secure_filename(database_name)
        
        # Create database-specific upload directory
        upload_dir = f"/data1/datatalk_domains/uploads/{database_name}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create a symbolic link for easier access
        symlink_path = f"/data1/datatalk_domains/{database_name}"
        try:
            # Remove existing symlink if it exists
            if os.path.islink(symlink_path):
                os.unlink(symlink_path)
            # Create the symbolic link
            os.symlink(upload_dir, symlink_path)
            print(f"Created symbolic link: {symlink_path} -> {upload_dir}")
        except Exception as e:
            print(f"Warning: Could not create symbolic link: {str(e)}")
        
        uploaded_files = []
        
        # Process each uploaded file
        files = request.files.getlist("files")
        
        if not files or all(not file.filename for file in files):
            return jsonify({"success": False, "error": "No files were uploaded"})
        
        for file_obj in files:
            if not file_obj.filename:
                continue
                
            file_name = file_obj.filename
            if not is_allowed_file(file_name):
                return jsonify({"success": False, "error": f"Invalid file extension for {file_name}. Allowed extensions are {ALLOWED_EXTENSIONS}."})
            
            # Create a temporary file to store the uploaded content
            temp_file = tempfile.NamedTemporaryFile(dir=upload_dir, delete=False)
            temp_file_path = temp_file.name
            
            try:
                # Save the file to the temporary path
                file_obj.save(temp_file_path)
                
                # Check file size
                file_size = os.path.getsize(temp_file_path)
                if file_size > MAX_FILE_SIZE:
                    return jsonify({"success": False, "error": f"File size exceeds the limit. Maximum size is {MAX_FILE_SIZE / (1024 * 1024 * 1024)} GB."})
                
                # Generate a secure file name and move the file to the final location
                try:
                    final_file_path = move_file_to_final_location(file_name, temp_file_path, database_name)
                    uploaded_files.append(final_file_path)
                except Exception as e:
                    return jsonify({"success": False, "error": f"File upload failed for {file_name}: {str(e)}"})
            except Exception as e:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return jsonify({"success": False, "error": f"File upload failed for {file_name}: {str(e)}"})
            finally:
                # Ensure the temporary file is removed if it still exists
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        if not uploaded_files:
            return jsonify({"success": False, "error": "No valid files were uploaded"})
        
        # Return JSON response for the frontend
        return jsonify({"success": True, "files": uploaded_files, "database": database_name})
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"})

def move_file_to_final_location(filename: str, temp_file_path: str, database_name: str) -> str:
    """Move the temporary file to the final location with a secure name."""
    secure_filename = generate_secure_filename(filename)
    
    # Use the database-specific directory in /data1/datatalk_domains/uploads
    directory = f"/data1/datatalk_domains/uploads/{database_name}"
    
    os.makedirs(directory, exist_ok=True)
    file_location = f"{directory}/{secure_filename}"

    # Always overwrite existing files
    if os.path.exists(file_location):
        os.remove(file_location)
    
    # Move the file to the final location
    shutil.move(temp_file_path, file_location)
    return file_location

def generate_secure_filename(filename: str) -> str:
    """Generate a secure filename by removing unsafe characters."""
    return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", filename).replace(" ", "_")

@app.route('/ingestion', methods=['POST'])
def ingestion():
    """
    Process uploaded CSV files with streaming response to prevent timeout.
    Uses a thread-based approach to stream output in real-time.
    """
    # Create a queue for inter-thread communication
    output_queue = queue.Queue()
    processing_done = threading.Event()
    
    # Capture request data before starting the thread
    # This must happen in the request context
    try:
        request_data = request.json
        database_name = request_data.get('database')
        files = request_data.get('files', [])
        
        if not database_name or not files:
            return "Error: Missing database name or files", 400
    except Exception as e:
        return f"Error parsing request: {str(e)}", 400
    
    def generate():
        """Generator function that yields output as it becomes available in the queue."""
        yield "Starting ingestion process...\n"
        
        # Continue yielding until processing is done and queue is empty
        while not (processing_done.is_set() and output_queue.empty()):
            try:
                # Get output from queue with timeout to prevent blocking forever
                # This also allows us to check the processing_done flag periodically
                message = output_queue.get(timeout=0.5)
                yield message
                output_queue.task_done()
            except queue.Empty:
                # No output available, but process might still be running
                if not processing_done.is_set():
                    # Send a periodic "heartbeat" to keep the connection alive
                    yield ""  # Empty string to keep connection alive
                    time.sleep(0.5)
        
    def output_writer(text):
        """Function to write output to the queue for streaming."""
        output_queue.put(text)
    
    def process_files_thread(database_name, files):
        """Thread function to process files and capture output."""
        try:
            output_writer(f"Processing files for database: {database_name}\n")
            
            # Extract filenames from paths
            file_paths = files
            
            # Define the upload directory for this database
            upload_dir = f"/data1/datatalk_domains/uploads/{database_name}"
            
            # Ensure the directory exists
            os.makedirs(upload_dir, exist_ok=True)
            
            # Create a symbolic link for easier access
            symlink_path = f"/data1/datatalk_domains/{database_name}"
            try:
                # Remove existing symlink if it exists
                if os.path.islink(symlink_path):
                    os.unlink(symlink_path)
                # Create the symbolic link
                os.symlink(upload_dir, symlink_path)
                output_writer(f"Created symbolic link: {symlink_path} -> {upload_dir}\n")
            except Exception as e:
                output_writer(f"Warning: Could not create symbolic link: {str(e)}\n")
            
            # Check if datatalk_declaration.csv exists among the files
            declaration_file = None
            csv_files = []
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                if filename == "datatalk_declaration.csv":
                    declaration_file = file_path
                elif filename.endswith('.csv'):
                    csv_files.append(file_path)
            
            output_writer(f"Found {len(csv_files)} CSV files and {'a' if declaration_file else 'no'} declaration file\n")
            
            # Create custom stdout/stderr streams that write to our queue
            class StreamToQueue(io.StringIO):
                def write(self, text):
                    super().write(text)
                    # Don't filter out whitespace-only lines (could be newlines)
                    # Instead send all non-empty content
                    if text:
                        # Preserve exact formatting including newlines
                        output_writer(text)
            
            # For handling the declaration file case
            if declaration_file:
                # We need to modify the declaration file to use full paths
                try:
                    # Read the declaration file
                    with open(declaration_file, 'r') as f:
                        declaration_lines = f.readlines()
                    
                    # The first line is the header
                    header = declaration_lines[0]
                    
                    # Find the indices of relevant columns
                    header_parts = header.strip().split(',')
                    csv_filepath_index = header_parts.index('csv_filepath') if 'csv_filepath' in header_parts else 0
                    csv_filepath_header_index = header_parts.index('csv_filepath_header') if 'csv_filepath_header' in header_parts else 1
                    
                    # Create a temporary modified declaration file
                    modified_declaration = os.path.join(upload_dir, "modified_datatalk_declaration.csv")
                    
                    with open(modified_declaration, 'w') as f:
                        # Write the header
                        f.write(header)
                        
                        # For each data line, modify the paths to be full paths
                        for line in declaration_lines[1:]:
                            line = line.strip()
                            if not line:  # Skip empty lines
                                f.write('\n')
                                continue
                                
                            parts = line.split(',')
                            
                            # Only process if we have enough parts
                            if len(parts) > max(csv_filepath_index, csv_filepath_header_index):
                                # Fix csv_filepath if it's not empty
                                if csv_filepath_index < len(parts) and parts[csv_filepath_index].strip():
                                    # Just use the standard path
                                    parts[csv_filepath_index] = os.path.join(upload_dir, parts[csv_filepath_index].strip())
                                
                                # Fix csv_filepath_header if it's not empty
                                if csv_filepath_header_index < len(parts) and parts[csv_filepath_header_index].strip():
                                    # Just use the standard path
                                    parts[csv_filepath_header_index] = os.path.join(upload_dir, parts[csv_filepath_header_index].strip())
                                
                                # Write the modified line
                                f.write(','.join(parts) + '\n')
                            else:
                                # If we can't parse the line properly, write it as is
                                f.write(line + '\n')
                    
                    # Use the modified declaration file
                    first_arg = modified_declaration
                    
                    output_writer(f"Using modified declaration file: {modified_declaration}\n")
                    output_writer("Modified declaration contains absolute paths to CSV files.\n")
                    
                except Exception as e:
                    output_writer(f"Error modifying declaration file: {str(e)}\n")
                    output_writer("Falling back to original declaration file.\n")
                    first_arg = declaration_file
            else:
                # If no declaration file, just use the list of CSV files
                output_writer("\n===== NOTICE =====\n")
                output_writer("No 'datatalk_declaration.csv' file was found among the uploaded files.\n")
                output_writer("If you intended to use a declaration file for table descriptions and column mappings, please upload one named 'datatalk_declaration.csv'.\n")
                output_writer("Proceeding with direct CSV ingestion using automatic type inference.\n")
                output_writer("==================\n\n")
                first_arg = csv_files
            
            # Set up custom streams
            stdout_stream = StreamToQueue()
            stderr_stream = StreamToQueue()
            
            # Process files and redirect output to our streams
            output_writer("Starting ingestion process with process_all_csvs...\n")
            
            try:
                # This is where the actual processing happens
                with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                    result = process_all_csvs(
                        first_arg,
                        database_name,
                        None,  # third argument
                        False,  # fourth argument
                        auto_create_db=True  # fifth argument
                    )
                
                # Capture any remaining output from the streams
                stdout_remaining = stdout_stream.getvalue()
                stderr_remaining = stderr_stream.getvalue()
                
                # Send any output that wasn't already streamed
                if stdout_remaining:
                    output_writer(stdout_remaining)
                
                # Process stderr - filter out incomplete progress bars but keep actual errors
                if stderr_remaining and stderr_remaining.strip():
                    # Common progress bar patterns to filter out
                    progress_bar_patterns = [
                        r'^\s*\d+%\|[^\n]*\[\d+:\d+.*$',      # TQDM progress bar format
                        r'^\s*\d+%\|[^\n]*\|\s*\d+/\d+.*$',   # Another common format
                        r'^\s*\d+%.*\d+/\d+.*$'               # More generic progress indication
                    ]
                    
                    # Process the stderr content line by line
                    stderr_lines = stderr_remaining.strip().split('\n')
                    filtered_stderr = []
                    
                    # Process each line to keep errors but filter progress bars
                    for line in stderr_lines:
                        # Check if this line is a progress bar
                        is_progress_bar = False
                        for pattern in progress_bar_patterns:
                            if re.match(pattern, line.strip()):
                                is_progress_bar = True
                                break
                        
                        # Keep this line if it's not a progress bar (even if empty)
                        if not is_progress_bar:
                            filtered_stderr.append(line)
                    
                    # If we have any error lines after filtering, show them
                    if filtered_stderr:
                        # Make sure we have a clear separator for error highlighting in frontend
                        output_writer("\n\n--- ERRORS ---\n")
                        # Join with newlines to preserve formatting
                        output_writer('\n'.join(filtered_stderr))
                        # Ensure there's a trailing newline
                        if not filtered_stderr[-1].endswith('\n'):
                            output_writer('\n')
                        # Add explicit error marker for frontend detection
                        output_writer("Error detected during processing.\n")
            
            except Exception as e:
                error_msg = f"\nError during processing: {str(e)}\n"
                traceback_text = traceback.format_exc()
                
                # Clearly mark errors for frontend highlighting
                output_writer("\n\n--- ERRORS ---\n")
                output_writer(error_msg)
                output_writer(traceback_text)
                output_writer("Error detected during processing.\n")
            
        except Exception as e:
            # Handle any unexpected errors
            output_writer(f"\n\nUnexpected error: {str(e)}\n")
            output_writer(traceback.format_exc())
        
        finally:
            # Signal that processing is done
            processing_done.set()
    
    # Start the processing thread with data from the request
    threading.Thread(target=process_files_thread, args=(database_name, files), daemon=True).start()
    
    # Create a streaming response
    response = Response(stream_with_context(generate()), mimetype='text/plain')
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering for nginx
    response.headers['Cache-Control'] = 'no-cache'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=False) 