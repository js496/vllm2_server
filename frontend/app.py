import os
import requests
import json
import logging
import ast
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
import redis
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
current_models_data: List[Dict] = []
rx_change_arr: List[int] = []

# Redis connection
try:
    r = redis.Redis(host="redis", port=6379, db=0)
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")

# Constants
CONTAINER_PORT = int(os.getenv("CONTAINER_PORT", 7860))
BASE_URL = f"http://container_backend:{CONTAINER_PORT + 1}/dockerrest"

def handle_errors(func):
    """Decorator to handle errors and log them."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {e}")
            return None
    return wrapper

@handle_errors
def get_gpu_data() -> List[Dict]:
    """Fetch GPU data from Redis."""
    gpu_data = r.get('db_gpu')
    return json.loads(gpu_data) if gpu_data else []

@handle_errors
def get_docker_container_list() -> List[Dict]:
    """Fetch the list of Docker containers."""
    response = requests.post(BASE_URL, json={"req_method": "list"})
    return response.json() if response.status_code == 200 else []

@handle_errors
def docker_api_logs(req_model: str) -> str:
    """Fetch logs for a specific Docker container."""
    response = requests.post(BASE_URL, json={"req_method": "logs", "req_model": req_model})
    return response.json().get("result_data", "") if response.status_code == 200 else "Error fetching logs"

@handle_errors
def docker_api_network(req_container_name: str) -> str:
    """Fetch network information for a specific Docker container."""
    response = requests.post(BASE_URL, json={"req_method": "network", "req_container_name": req_container_name})
    if response.status_code == 200:
        return response.json().get("result_data", {}).get("networks", {}).get("eth0", {}).get("rx_bytes", "")
    return "Error fetching network information"

@handle_errors
def docker_api_start(req_model: str) -> Dict:
    """Start a Docker container."""
    response = requests.post(BASE_URL, json={"req_method": "start", "req_model": req_model})
    return response.json() if response.status_code == 200 else {"error": "Failed to start container"}

@handle_errors
def docker_api_stop(req_model: str) -> Dict:
    """Stop a Docker container."""
    response = requests.post(BASE_URL, json={"req_method": "stop", "req_model": req_model})
    return response.json() if response.status_code == 200 else {"error": "Failed to stop container"}

@handle_errors
def docker_api_delete(req_model: str) -> Dict:
    """Delete a Docker container."""
    response = requests.post(BASE_URL, json={"req_method": "delete", "req_model": req_model})
    return response.json() if response.status_code == 200 else {"error": "Failed to delete container"}

@handle_errors
def docker_api_create(req_model: str, req_pipeline_tag: str, req_port_model: int, req_port_vllm: int) -> Dict:
    """Create a Docker container."""
    req_container_name = str(req_model).replace('/', '_')
    response = requests.post(BASE_URL, json={
        "req_method": "create",
        "req_container_name": req_container_name,
        "req_model": req_model,
        "req_runtime": "nvidia",
        "req_port_model": req_port_model,
        "req_port_vllm": req_port_vllm
    })
    if response.status_code == 200:
        new_entry = [{
            "gpu": 8,
            "path": f'/home/cloud/.cache/huggingface/{req_model}',
            "container": "0",
            "container_status": "0",
            "running_model": req_container_name,
            "model": req_model,
            "pipeline_tag": req_pipeline_tag,
            "port_model": req_port_model,
            "port_vllm": req_port_vllm
        }]
        r.set("db_gpu", json.dumps(new_entry))
        return response.json()
    return {"error": "Failed to create container"}

@handle_errors
def search_models(query: str) -> Dict:
    """Search for models on Hugging Face."""
    response = requests.get(f"https://huggingface.co/api/models?search={query}")
    if response.status_code == 200:
        models = response.json()
        model_ids = [m["id"] for m in models]
        return {"choices": model_ids, "value": models[0]["id"], "label": f"Found {len(models)} models!"}
    return {"error": "Failed to search models"}

@handle_errors
def calculate_model_size(json_info: Dict) -> int:
    """
    Calculate the size of a model based on its configuration JSON.

    Args:
        json_info (Dict): Model configuration JSON.

    Returns:
        int: Total size of the model in bytes.
    """
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0

        # Calculate size for Vision Transformer (ViT) models
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        # Calculate embedding size
        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        # Calculate attention and FFN weights size
        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)
            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        # Calculate size for encoder-decoder models
        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes * bytes_per_param

    except Exception as e:
        logger.error(f"Error calculating model size: {e}")
        return 0

@handle_errors
def get_info(selected_id: str) -> Tuple[Dict, str, str, bool, bool, int, str]:
    """
    Retrieve information about a specific model from the current models data.

    Args:
        selected_id (str): The ID of the model to retrieve information for.

    Returns:
        Tuple: Contains search data, model ID, pipeline tag, transformers flag, private flag, downloads, and container name.
    """
    res_model_data = {
        "search_data": {},
        "model_id": selected_id,
        "pipeline_tag": "",
        "transformers": False,
        "private": False,
        "downloads": 0,
    }

    try:
        for item in current_models_data:
            if item["id"] == selected_id:
                res_model_data["search_data"] = item
                res_model_data["pipeline_tag"] = item.get("pipeline_tag", "")
                res_model_data["transformers"] = "transformers" in item.get("tags", [])
                res_model_data["private"] = item.get("private", False)
                res_model_data["downloads"] = item.get("downloads", 0)
                container_name = str(selected_id).replace("/", "_")
                return (
                    res_model_data["search_data"],
                    res_model_data["model_id"],
                    res_model_data["pipeline_tag"],
                    res_model_data["transformers"],
                    res_model_data["private"],
                    res_model_data["downloads"],
                    container_name,
                )
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], ""

@handle_errors
def get_additional_info(selected_id: str) -> Tuple[Dict, Dict, str, int, bool]:
    """
    Retrieve additional information about a model from Hugging Face Hub.

    Args:
        selected_id (str): The ID of the model to retrieve additional information for.

    Returns:
        Tuple: Contains Hugging Face data, config data, model ID, size, and gated flag.
    """
    res_model_data = {
        "hf_data": {},
        "config_data": {},
        "model_id": selected_id,
        "size": 0,
        "gated": False,
    }

    try:
        # Fetch model info from Hugging Face Hub
        model_info = huggingface_hub.model_info(selected_id)
        model_info_json = vars(model_info)
        res_model_data["hf_data"] = model_info_json
        res_model_data["gated"] = model_info_json.get("gated", False)

        # Fetch model size from safetensors if available
        if "safetensors" in model_info_json:
            safetensors_json = vars(model_info.safetensors)
            res_model_data["size"] = safetensors_json.get("total", 0)

        # Fetch config.json from Hugging Face
        config_url = f"https://huggingface.co/{selected_id}/resolve/main/config.json"
        response = requests.get(config_url)
        if response.status_code == 200:
            res_model_data["config_data"] = response.json()
        else:
            res_model_data["config_data"] = f"Error: {response.status_code}"

        # Calculate model size if not already available
        if res_model_data["size"] == 0:
            res_model_data["size"] = calculate_model_size(res_model_data["config_data"])

        return (
            res_model_data["hf_data"],
            res_model_data["config_data"],
            res_model_data["model_id"],
            res_model_data["size"],
            res_model_data["gated"],
        )

    except Exception as e:
        logger.error(f"Error retrieving additional model info: {e}")
        return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]

@handle_errors
def gr_load_check(
    selected_model_id: str,
    selected_model_pipeline_tag: str,
    selected_model_transformers: bool,
    selected_model_private: bool,
    selected_model_gated: bool,
) -> Tuple[gr.update, gr.update]:
    """
    Check if the model can be loaded and update Gradio components accordingly.

    Args:
        selected_model_id (str): The ID of the selected model.
        selected_model_pipeline_tag (str): The pipeline tag of the selected model.
        selected_model_transformers (bool): Whether the model is a transformers model.
        selected_model_private (bool): Whether the model is private.
        selected_model_gated (bool): Whether the model is gated.

    Returns:
        Tuple: Gradio updates for the info textbox and download button.
    """
    if (
        selected_model_pipeline_tag
        and selected_model_transformers
        and not selected_model_private
        and not selected_model_gated
    ):
        return gr.update(visible=False), gr.update(value=f"Download {selected_model_id[:8]}...", visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

@handle_errors
def check_rx_change(current_rx_bytes: str) -> str:
    """
    Check for changes in received bytes to monitor download progress.

    Args:
        current_rx_bytes (str): Current received bytes as a string.

    Returns:
        str: Status message indicating download progress.
    """
    try:
        global rx_change_arr
        current_rx_bytes_int = int(current_rx_bytes)
        rx_change_arr.append(current_rx_bytes_int)

        if len(rx_change_arr) > 4:
            last_value = rx_change_arr[-1]
            same_value_count = sum(1 for val in rx_change_arr[-10:] if val == last_value)

            if same_value_count > 10:
                return "Count > 10: Download finished"
            else:
                return f"Count: {same_value_count} {rx_change_arr}"

        return f"Count: {len(rx_change_arr)} {rx_change_arr}"

    except ValueError:
        return "0"
    except Exception as e:
        logger.error(f"Error checking RX change: {e}")
        return f"Error: {e}"

@handle_errors
def json_to_pd() -> pd.DataFrame:
    """
    Convert GPU data from JSON to a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing GPU information.
    """
    rows = []
    try:
        gpu_list = get_gpu_data()
        for entry in gpu_list:
            gpu_info = ast.literal_eval(entry["gpu_info"])[0]
            rows.append({
                "#": entry["gpu"],
                "GPU Usage": f'{gpu_info["gpu_util"]} %',
                "Memory Usage": f'{gpu_info["mem_util"]:.2f} % ({gpu_info["mem_used"]} MB/{gpu_info["mem_total"]} MB)',
                "Running model": entry["running_model"],
                "Timestamp": entry["timestamp"],
                "Port vLLM": entry["port_vllm"],
                "Port model": entry["port_model"],
                "Used ports": entry["used_ports"],
                "Used models": entry["used_models"],
            })
        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"Error converting JSON to DataFrame: {e}")
        return pd.DataFrame(rows)


with gr.Blocks() as app:
    gr.Markdown("# Welcome! Select a Hugging Face model or tag and deploy it on different ports.")
    
    inp = gr.Textbox(placeholder="Type in a Hugging Face model or tag", show_label=False, autofocus=True)
    btn = gr.Button("Search")
    model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False, visible=False)
    
    with gr.Row():
        selected_model_id = gr.Textbox(label="id", visible=True)
        selected_model_container_name = gr.Textbox(label="container_name", visible=True)
        
    with gr.Row():       
        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag", visible=False)
        selected_model_transformers = gr.Textbox(label="transformers", visible=False)
        selected_model_private = gr.Textbox(label="private", visible=False)
        
    with gr.Row():
        selected_model_size = gr.Textbox(label="size", visible=False)
        selected_model_gated = gr.Textbox(label="gated", visible=False)
        selected_model_downloads = gr.Textbox(label="downloads", visible=False)
    
    selected_model_search_data = gr.Textbox(label="search_data", visible=False)
    selected_model_hf_data = gr.Textbox(label="hf_data", visible=False)
    selected_model_config_data = gr.Textbox(label="config_data", visible=False)
    gr.Markdown(
        """
        <hr>
        """
    )            
        
    inp.submit(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)
    btn.click(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)

    with gr.Row():
        port_model = gr.Number(value=8001,visible=False,label="Port of model: ")
        port_vllm = gr.Number(value=8000,visible=False,label="Port of vLLM: ")
    
    info_textbox = gr.Textbox(value="Interface not possible for selected model. Try another model or check 'pipeline_tag', 'transformers', 'private', 'gated'", show_label=False, visible=False)
    btn_dl = gr.Button("Download", visible=False)
    
    model_dropdown.change(get_info, model_dropdown, [selected_model_search_data,selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads],selected_model_container_name).then(get_additional_info, model_dropdown, [selected_model_hf_data, selected_model_config_data, selected_model_id, selected_model_size, selected_model_gated]).then(lambda: gr.update(visible=True), None, selected_model_pipeline_tag).then(lambda: gr.update(visible=True), None, selected_model_transformers).then(lambda: gr.update(visible=True), None, selected_model_private).then(lambda: gr.update(visible=True), None, selected_model_downloads).then(lambda: gr.update(visible=True), None, selected_model_size).then(lambda: gr.update(visible=True), None, selected_model_gated).then(lambda: gr.update(visible=True), None, port_model).then(lambda current_value: current_value + 1, port_model, port_model).then(lambda: gr.update(visible=True), None, port_vllm).then(lambda current_value: current_value + 1, port_vllm, port_vllm).then(gr_load_check, [selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated],[info_textbox,btn_dl])

    create_response = gr.Textbox(visible=False,label="Building container...", show_label=True)  

    btn_interface = gr.Button("Load Interface",visible=False)
    @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
    def show_split(text_pipeline, text_model):
        if len(text_model) == 0:
            gr.Markdown("Error pipeline_tag or model_id")
        else:
            gr.Interface.from_pipeline(pipeline(text_pipeline, model=text_model))

    gpu_dataframe = gr.Dataframe(label="GPU information")
    gpu_timer = gr.Timer(1,active=True)
    gpu_timer.tick(json_to_pd, outputs=gpu_dataframe)
    container_state = gr.State([])   
    docker_container_list = get_docker_container_list()     
    @gr.render(inputs=container_state)
    def render_container(render_container_list):
        docker_container_list = get_docker_container_list()
        docker_container_list_running = [c for c in docker_container_list if c["State"]["Status"] == "running"]
        docker_container_list_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running"]

        def refresh_container():
            try:
                global docker_container_list
                response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                docker_container_list = response.json()
                return docker_container_list
            
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'
            
        gr.Markdown(f'### Container running ({len(docker_container_list_running)})')

        for current_container in docker_container_list_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

            with gr.Row():            
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)     
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )

                stop_btn = gr.Button("Stop", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                stop_btn.click(
                    docker_api_stop,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
                
            gr.Markdown(
                """
                <hr>
                """
            )


        gr.Markdown(f'### Container not running ({len(docker_container_list_not_running)})')

        for current_container in docker_container_list_not_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
            
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                
            with gr.Row():
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                                
                start_btn = gr.Button("Start", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                start_btn.click(
                    docker_api_start,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
            
            gr.Markdown(
                """
                <hr>
                """
            )
            
    def refresh_container_list():
        try:
            global docker_container_list
            response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
            docker_container_list = response.json()
            return docker_container_list
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return f'err {str(e)}'
                 
    def check_container_running(container_name):
        try:
            docker_container_list = get_docker_container_list()
            docker_container_list_running = [c for c in docker_container_list if c["name"] == container_name]
            if len(docker_container_list) > 0:
                return f'Yes container is running!'
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return f'err {str(e)}'
    
    timer_dl = gr.Timer(1,active=False)
    timer_dl.tick(docker_api_network, selected_model_container_name, create_response)
    
    btn_dl.click(lambda: gr.update(label="Download Progress:",visible=True), None, create_response).then(docker_api_create,inputs=[model_dropdown,selected_model_pipeline_tag,port_model,port_vllm],outputs=create_response).then(refresh_container_list, outputs=[container_state]).then(lambda: gr.Timer(active=True), None, timer_dl).then(lambda: gr.update(visible=True), None, btn_interface)

app.launch(server_name="0.0.0.0", server_port=int(os.getenv("CONTAINER_PORT")))
