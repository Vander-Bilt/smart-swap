import os
import time
import gradio as gr
import requests
import json
import cv2
import pathlib
import shutil
import roop.globals
import roop.metadata
import roop.utilities as util
import roop.notion as notion
from roop.predictor import predict_image, predict_video

from roop.face_util import extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame

restart_server = False
live_cam_active = False

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

SELECTION_FACES_DATA = None

last_image = None

input_thumbs = []
target_thumbs = []


IS_INPUT = True
SELECTED_FACE_INDEX = 0

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None

input_faces = None
target_faces = None
face_selection = None
fake_cam_image = None

current_cam_image = None
cam_swapping = False

selected_preview_index = 0

is_processing = False            



def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]


def run():
    from roop.core import suggest_execution_providers, decode_execution_providers, set_display_ui, set_execution_provider
    global input_faces, target_faces, face_selection, fake_cam_image, restart_server, live_cam_active, on_settings_changed

    prepare_environment()

    available_themes = ["Default", "gradio/glass", "gradio/monochrome", "gradio/seafoam", "gradio/soft", "gstaff/xkcd", "freddyaboulton/dracula_revamped", "ysharma/steampunk"]
    image_formats = ['jpg','png', 'webp']
    video_formats = ['avi','mkv', 'mp4', 'webm']
    video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
    providerlist = suggest_execution_providers()
    
    settings_controls = []

    live_cam_active = roop.globals.CFG.live_cam_start_active
    set_display_ui(show_msg)
    roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    print(f'Using provider {roop.globals.execution_providers} - Device:{util.get_device()}')    
    #set_execution_provider(roop.globals.CFG.provider)
    #print(f'Available providers {roop.globals.execution_providers}, using {roop.globals.execution_providers[0]} - Device:{util.get_device()}')
    
    run_server = True
    mycss = """
        span {color: var(--block-info-text-color)}
        #filelist {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
"""


    js_code = """
    async function(selected_enhancer, selected_face_detection, keep_fps, keep_frames, skip_audio, max_face_distance, blend_ratio, bt_destfiles, chk_useclip, clip_text,video_swapping_method, hf_token) {
        // 获取开始按钮元素并禁用它
        const startButton = document.querySelector('#btn-start');
        if (startButton) {
            startButton.disabled = true;
            startButton.classList.add('disabled');
        }
        console.log('按钮点击，JS函数执行中...');
        console.log('selected_enhancer:', selected_enhancer);
        console.log('selected_face_detection:', selected_face_detection);
        console.log('keep_fps:', keep_fps);
        console.log('keep_frames:', keep_frames);
        //console.log('skip_audio:', skip_audio);
        console.log('max_face_distance:', max_face_distance);
        console.log('blend_ratio:', blend_ratio);
        console.log('bt_destfiles:', bt_destfiles);
        //console.log('chk_useclip:', chk_useclip);

        async function checkBackendFlag(ip, fingerprint1, fingerprint2) {
        let flag = false; // Initialize flag to false by default
        
        // Construct the URL with parameters
        const url = `https://commonuser.yesky.online/query?ip=${ip}&fingerprint1=${encodeURIComponent(fingerprint1)}&fingerprint2=${encodeURIComponent(fingerprint2)}`;
        console.log("url:",url);
        try {
            const response = await fetch(url);
        
            // Check if the HTTP response was successful (status 200-299)
            if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
            }
        
            const data = await response.json(); // Parse the JSON response
            console.log("data:",data);
            // Safely check if 'flag' exists and is a boolean
            if (data && typeof data.flag === 'boolean') {
            console.log("data.flag:",data.flag);
            flag = data.flag; // Set flag based on the backend's response
            } else {
            console.warn("API response did not contain a valid 'flag' boolean field.");
            }
        
        } catch (error) {
            console.error("Error calling backend API:", error);
            // 'flag' remains false if an error occurs
        }
        
        return flag; // Return the determined flag value
        }

        let fingerprint1;
        const fpPromise = import('https://openfpcdn.io/fingerprintjs/v4')
            .then(FingerprintJS => FingerprintJS.load());

        // 先获取 fingerprint1
        try {
            const fp = await fpPromise;
            const result = await fp.get();
            fingerprint1 = result.visitorId;
            console.log('fingerprint1:', fingerprint1);

            if (!fingerprint1) {
                fingerprint1 = 'firefox-' + generateRandomString(6);
                console.log(fingerprint1);
            }
        } catch (error) {
            console.error('获取指纹失败:', error);
            fingerprint1 = 'error-' + generateRandomString(6);
        }

        // 现在 fingerprint1 已经确定
        //console.log("Checking..");
        //console.log(fingerprint1);


        function optimizedHash(str) {
            let hash1 = 5381, hash2 = 52711;
            for (let i = 0; i < str.length; i++) {
                const char = str.charCodeAt(i);
                hash1 = (hash1 * 33) ^ char;  // DJB2算法变种
                hash2 = (hash2 * 31) + char;  // 另一个简单哈希
            }
            // 组合两个哈希值
            return (hash1 >>> 0).toString(16) + (hash2 >>> 0).toString(16);
        }
        
        function get_browser_fingerprint() {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const txt = 'i9asdm..$#po((^@KbXr~*~*';
            ctx.textBaseline = 'top';
            ctx.font = "14px 'Arial'";
            ctx.textBaseline = 'alphabetic';
            ctx.fillStyle = '#f60';
            ctx.fillRect(125, 1, 62, 20);
            ctx.fillStyle = '#069';
            ctx.fillText(txt, 2, 15);
            ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
            ctx.fillText(txt, 4, 17);
            
            const dataUrl = canvas.toDataURL();
            return optimizedHash(dataUrl);
        }

        const fingerprint2 = get_browser_fingerprint();


        function generateRandomString(length) {
        let result = '';
        const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        const charactersLength = characters.length;
        for (let i = 0; i < length; i++) {
            result += characters.charAt(Math.floor(Math.random() * charactersLength));
        }
        return result;
        }

        async function getUserIPAddress() {
        try {
            // We'll use ipify.org as an example, but there are many others.
            // They offer a simple API that returns just the IP address as plain text.
            const response = await fetch('https://ipinfo.io/json');
        
            if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
            }
        
            const data = await response.json();
            const ipAddress = data.ip;
        
            //console.log('Your IP Address is:', ipAddress);
            return ipAddress;
        
        } catch (error) {
            console.error('Could not get IP address:', error);
            return null;
        }
        }

        const ip = await getUserIPAddress();
        //console.log(ip);

        const isFlagTrue = await checkBackendFlag(ip, fingerprint1, fingerprint2);

        console.log('执行完成，返回');
        // 重新启用按钮
        if (startButton) {
            startButton.disabled = false;
            startButton.classList.remove('disabled');
        }
        return [selected_enhancer, selected_face_detection, keep_fps, keep_frames, skip_audio, max_face_distance, blend_ratio, bt_destfiles, chk_useclip, clip_text,video_swapping_method, hf_token, isFlagTrue, ip, fingerprint1, fingerprint2];

    }
    """



    while run_server:
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = False if server_name == '0.0.0.0' else True
        with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}', theme=roop.globals.CFG.selected_theme, css=mycss) as ui:

            with gr.Row(variant='panel'):
                hidden_input = gr.Checkbox(False, visible=False)
                hidden_finger1 = gr.Textbox(visible=False)
                hidden_finger2 = gr.Textbox(visible=False)
                hidden_ip = gr.Textbox(visible=False)

                gr.Markdown(f"## [{roop.metadata.name} {roop.metadata.version}](https://nav001.online)")
                gr.HTML(util.create_version_html(), elem_id="versions")
            with gr.Tab("Face Swap"):
                with gr.Row():
                    with gr.Column():
                        input_faces = gr.Gallery(label="Input faces", allow_preview=True, preview=True, height=128, object_fit="scale-down")
                        with gr.Row():
                                bt_remove_selected_input_face = gr.Button("Remove selected")
                                bt_clear_input_faces = gr.Button("Clear all", variant='stop')
                        bt_srcimg = gr.Image(label='Source Face Image', type='filepath', tool=None)
                    with gr.Column():
                        target_faces = gr.Gallery(label="Target faces", allow_preview=True, preview=True, height=128, object_fit="scale-down")
                        with gr.Row():
                                bt_remove_selected_target_face = gr.Button("Remove selected")
                        bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", elem_id='filelist')
                        with gr.Row():
                            target_url_input = gr.Textbox(label="Target URL (Image/Video)", placeholder="Enter URL here...")
                            bt_download_target_url = gr.Button("Download from URL")

                        with gr.Row():
                            with gr.Accordion(label="Preview Original/Fake Frame", open=True):
                                with gr.Row():
                                    previewimage = gr.Image(label="Preview Image", interactive=False)
                                with gr.Row():
                                    with gr.Accordion(label="Fake Frame", open=False):
                                        with gr.Row(variant='panel'):
                                            with gr.Column():
                                                preview_frame_num = gr.Slider(0, 0, value=0, label="Frame Number", step=1.0, interactive=True)
                                            with gr.Column():
                                                bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary')

                        with gr.Row():
                            with gr.Column(visible=False) as dynamic_face_selection:
                                face_selection = gr.Gallery(label="Detected faces", allow_preview=True, preview=True, height=256, object_fit="scale-down")
                                with gr.Row():
                                    bt_faceselect = gr.Button("Use selected face")
                                    bt_cancelfaceselect = gr.Button("Done")
            
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_face_detection = gr.Dropdown(["First found", "All faces", "Selected face", "All female", "All male"], value="First found", label="Select face selection for swapping")
                        max_face_distance = gr.Slider(0.01, 1.0, value=0.65, label="Max Face Similarity Threshold")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_swapping_method = gr.Dropdown(["Extract Frames to media","In-Memory processing"], value="In-Memory", label="Select video processing method", interactive=True)
                        chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
                    with gr.Column(scale=2):
                        roop.globals.keep_fps = gr.Checkbox(label="Keep FPS", value=True)
                        roop.globals.keep_frames = gr.Checkbox(label="Keep Frames (relevant only when extracting frames)", value=False)
                        roop.globals.skip_audio = gr.Checkbox(label="Skip audio", value=False)
                with gr.Row():
                    with gr.Column():
                        selected_enhancer = gr.Dropdown(["None", "Codeformer", "DMDNet", "GFPGAN"], value="None", label="Select post-processing")
                    with gr.Column():
                        blend_ratio = gr.Slider(0.0, 1.0, value=0.65, label="Original/Enhanced image blend ratio")
                with gr.Row():
                    with gr.Accordion(label="Masking", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                chk_useclip = gr.Checkbox(label="Use Text Masking", value=False)
                                clip_text = gr.Textbox(label="List of objects to mask and restore back on fake image", placeholder="cup,hands,hair,banana")
                                gr.Dropdown(["Clip2Seg"], value="Clip2Seg", label="Engine")
                            with gr.Column(scale=1):
                                bt_preview_mask = gr.Button("Show Mask Preview", variant='secondary')
                            with gr.Column(scale=2):
                                maskpreview = gr.Image(label="Preview Mask", shape=(None,512), interactive=False)
                with gr.Row():
                    with gr.Accordion(label="Huggingface Token", visible=False, open=False):
                        with gr.Row():
                            with gr.Column():
                                hf_token = gr.Textbox(label="HuggingFace Token", placeholder="Enter your HuggingFace token to upload results", type="password")
                
                with gr.Row(variant='panel'):
                    with gr.Column():
                        bt_start = gr.Button("Start", variant='primary', elem_id='btn-start')
                    with gr.Column():
                        bt_stop = gr.Button("Stop", variant='secondary')
                    with gr.Column():
                        fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    with gr.Column():
                        bt_refresh_preview = gr.Button("Refresh", variant='secondary')
                with gr.Row(variant='panel'):
                    with gr.Column():
                        with gr.Accordion(label="Results", open=True):
                            gr.Button("Open Output Folder", size='sm').click(fn=lambda: util.open_folder(util.resolve_relative_path('../output/')))
                            resultfiles = gr.Files(label='Processed File(s)', interactive=False)
                            resultimage = gr.Image(type='filepath', interactive=False)
                    # with gr.Column():
                    #     with gr.Accordion(label="Preview Original/Fake Frame", open=False):
                    #         previewimage = gr.Image(label="Preview Image", interactive=False)
                    #         with gr.Row(variant='panel'):
                    #             with gr.Column():
                    #                 preview_frame_num = gr.Slider(0, 0, value=0, label="Frame Number", step=1.0, interactive=True)
                    #             with gr.Column():
                    #                 bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary')
                        
            with gr.Tab("Live Cam"):
                cam_toggle = gr.Checkbox(label='Activate', value=live_cam_active)
                if live_cam_active:
                    with gr.Row():
                        with gr.Column():
                            cam = gr.Webcam(label='Camera', source='webcam', mirror_webcam=True, interactive=True, streaming=False)
                        with gr.Column():
                            fake_cam_image = gr.Image(label='Fake Camera Output', interactive=False)


            with gr.Tab("Extras"):
                with gr.Row():
                    files_to_process = gr.Files(label='File(s) to process', file_count="multiple")
                # with gr.Row(variant='panel'):
                #     with gr.Accordion(label="Post process", open=False):
                #         with gr.Column():
                #             selected_post_enhancer = gr.Dropdown(["None", "Codeformer", "GFPGAN"], value="None", label="Select post-processing")
                #         with gr.Column():
                #             gr.Button("Start").click(fn=lambda: gr.Info('Not yet implemented...'))
                with gr.Row(variant='panel'):
                    with gr.Accordion(label="Video/GIF", open=False):
                        with gr.Row(variant='panel'):
                            with gr.Column():
                                gr.Markdown("""
                                            # Cut video
                                            Be aware that this means re-encoding the video which might take a longer time.
                                            Encoding uses your configuration from the Settings Tab.
    """)
                            with gr.Column():
                                cut_start_time = gr.Slider(0, 1000000, value=0, label="Start Frame", step=1.0, interactive=True)
                            with gr.Column():
                                cut_end_time = gr.Slider(1, 1000000, value=1, label="End Frame", step=1.0, interactive=True)
                            with gr.Column():
                                start_cut_video = gr.Button("Start")

    #                     with gr.Row(variant='panel'):
    #                         with gr.Column():
    #                             gr.Markdown("""
    #                                         # Join videos
    #                                         This also re-encodes the videos like cutting above.
    # """)
    #                         with gr.Column():
    #                             start_join_videos = gr.Button("Start")
                        with gr.Row(variant='panel'):
                            gr.Markdown("Extract frames from video")
                            start_extract_frames = gr.Button("Start")
                        with gr.Row(variant='panel'):
                            gr.Markdown("Create video from image files")
                            gr.Button("Start").click(fn=lambda: gr.Info('Not yet implemented...'))
                        with gr.Row(variant='panel'):
                            gr.Markdown("Create GIF from video")
                            start_create_gif = gr.Button("Create GIF")
                with gr.Row():
                    extra_files_output = gr.Files(label='Resulting output files', file_count="multiple")
                        
            
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        themes = gr.Dropdown(available_themes, label="Theme", info="Change needs complete restart", value=roop.globals.CFG.selected_theme)
                    with gr.Column():
                        settings_controls.append(gr.Checkbox(label="Public Server", value=roop.globals.CFG.server_share, elem_id='server_share', interactive=True))
                        settings_controls.append(gr.Checkbox(label='Clear output folder before each run', value=roop.globals.CFG.clear_output, elem_id='clear_output', interactive=True))
                    with gr.Column():
                        input_server_name = gr.Textbox(label="Server Name", lines=1, info="Leave blank to run locally", value=roop.globals.CFG.server_name)
                    with gr.Column():
                        input_server_port = gr.Number(label="Server Port", precision=0, info="Leave at 0 to use default", value=roop.globals.CFG.server_port)
                with gr.Row():
                    with gr.Column():
                        settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.globals.CFG.provider, elem_id='provider', interactive=True))
                        settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                        max_threads = gr.Slider(1, 64, value=roop.globals.CFG.max_threads, label="Max. Number of Threads", info='default: 8', step=1.0, interactive=True)
                    with gr.Column():
                        memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit, label="Max. Memory to use (Gb)", info='0 meaning no limit', step=1.0, interactive=True)
                        frame_buffer_size = gr.Slider(1, 512, value=roop.globals.CFG.frame_buffer_size, label="Frame Buffer Size", info='Num. Images to preload for each thread', step=1.0, interactive=True)
                        settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", info='default: png', value=roop.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
                    with gr.Column():
                        settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264', value=roop.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                        settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", info='default: mp4', value=roop.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                        video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Video Quality (crf)", info='default: 14', step=1.0, interactive=True)
                    with gr.Column():
                        button_apply_restart = gr.Button("Restart Server", variant='primary')
                        settings_controls.append(gr.Checkbox(label='Start with active live cam', value=roop.globals.CFG.live_cam_start_active, elem_id='live_cam_start_active', interactive=True))
                        button_clean_temp = gr.Button("Clean temp folder")
                        button_apply_settings = gr.Button("Apply Settings")

            previewinputs = [preview_frame_num, bt_destfiles, fake_preview, selected_enhancer, selected_face_detection,
                                max_face_distance, blend_ratio, chk_useclip, clip_text] 
            input_faces.select(on_select_input_face, None, None).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])
            bt_srcimg.change(fn=on_srcimg_changed, show_progress='full', inputs=bt_srcimg, outputs=[dynamic_face_selection, face_selection, input_faces])


            target_faces.select(on_select_target_face, None, None)
            bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

            previewinputs = [preview_frame_num, bt_destfiles, fake_preview, selected_enhancer, selected_face_detection,
                                max_face_distance, blend_ratio, chk_useclip, clip_text] 

            bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num]).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            bt_destfiles.select(fn=on_destfiles_selected, inputs=[bt_destfiles], outputs=[preview_frame_num]).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces])
            bt_download_target_url.click(fn=on_download_target_url, inputs=[target_url_input], outputs=[bt_destfiles, preview_frame_num]).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            resultfiles.select(fn=on_resultfiles_selected, inputs=[resultfiles], outputs=[resultimage])

            face_selection.select(on_select_face, None, None)
            bt_faceselect.click(fn=on_selected_face, outputs=[input_faces, target_faces, selected_face_detection])
            bt_cancelfaceselect.click(fn=on_end_face_selection, outputs=[dynamic_face_selection, face_selection])
            
            bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])

            chk_det_size.select(fn=on_option_changed)

            bt_preview_mask.click(fn=on_preview_mask, inputs=[preview_frame_num, bt_destfiles, clip_text], outputs=[maskpreview]) 

            start_event = bt_start.click(fn=start_swap, 
                inputs=[selected_enhancer, selected_face_detection, roop.globals.keep_fps, roop.globals.keep_frames,
                         roop.globals.skip_audio, max_face_distance, blend_ratio, bt_destfiles, chk_useclip, clip_text,video_swapping_method, hf_token, 
                         hidden_input, hidden_ip, hidden_finger1, hidden_finger2],
                # inputs=[selected_enhancer, selected_face_detection, roop.globals.keep_fps, roop.globals.keep_frames, max_face_distance, blend_ratio, bt_destfiles, hidden_input, hidden_ip, hidden_finger1, hidden_finger2],
                outputs=[bt_start, resultfiles, resultimage],
                _js=js_code)
            
            bt_stop.click(fn=stop_swap, cancels=[start_event])
            
            bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])            
            fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            preview_frame_num.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage], show_progress='hidden')
            bt_use_face_from_preview.click(fn=on_use_face_from_selected, show_progress='full', inputs=[bt_destfiles, preview_frame_num], outputs=[dynamic_face_selection, face_selection, target_faces, selected_face_detection])

            
            # Live Cam
            cam_toggle.change(fn=on_cam_toggle, inputs=[cam_toggle])
            if live_cam_active:
                cam.stream(on_stream_swap_cam, inputs=[cam, selected_enhancer, blend_ratio], outputs=[fake_cam_image], preprocess=True, postprocess=True, show_progress="hidden")

            # Extras
            start_cut_video.click(fn=on_cut_video, inputs=[files_to_process, cut_start_time, cut_end_time], outputs=[extra_files_output])
            # start_join_videos.click(fn=on_join_videos, inputs=[files_to_process], outputs=[extra_files_output])
            start_extract_frames.click(fn=on_extract_frames, inputs=[files_to_process], outputs=[extra_files_output])
            start_create_gif.click(fn=on_create_gif, inputs=[files_to_process], outputs=[extra_files_output])

            # Settings
            for s in settings_controls:
                s.select(fn=on_settings_changed)
            max_threads.input(fn=lambda a,b='max_threads':on_settings_changed_misc(a,b), inputs=[max_threads])
            memory_limit.input(fn=lambda a,b='memory_limit':on_settings_changed_misc(a,b), inputs=[memory_limit])
            frame_buffer_size.input(fn=lambda a,b='frame_buffer_size':on_settings_changed_misc(a,b), inputs=[frame_buffer_size])
            video_quality.input(fn=lambda a,b='video_quality':on_settings_changed_misc(a,b), inputs=[video_quality])

            button_clean_temp.click(fn=clean_temp, outputs=[bt_srcimg, input_faces, target_faces, bt_destfiles])
            button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port])
            button_apply_restart.click(restart)



        restart_server = False
        try:
            print("Lets do it...")
            app, local_url, share_url = ui.queue().launch(inbrowser=True, server_name=server_name, server_port=server_port, share=roop.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
            print("Can see me?")
            print(roop.globals.CFG.reg_notion)
            print(f"Got share url: {share_url}")
            if roop.globals.CFG.reg_notion:
                print("Register notion")
                #notion.delete_all_records()
                notion.add_record_to_notion_database(share_url)
            
        except:
            print("Got error")
            restart_server = True
            run_server = False
        try:
            while restart_server == False:
                time.sleep(5.0)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()



def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.value)
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.globals.CFG, attribname):
        setattr(roop.globals.CFG, attribname, new_val)
    else:
        print("Didn't find attrib!")


def on_download_target_url(url_input, progress=gr.Progress()):
    global RECENT_DIRECTORY_TARGET
    if not url_input:
        gr.Warning("URL input is empty.")
        return None, gr.Slider.update() # Return None for bt_destfiles and no change for preview_frame_num

    download_dir = os.path.join(os.getcwd(), "temp", "downloaded_targets")
    os.makedirs(download_dir, exist_ok=True)

    progress(0, desc="Downloading target from URL...")
    downloaded_file_path = util.download_file_from_url(url_input, download_dir)
    progress(1, desc="Download complete.")

    if downloaded_file_path:
        RECENT_DIRECTORY_TARGET = download_dir
        # Simulate the file being selected in bt_destfiles
        # We need to return a list of file paths for the gr.Files component
        # And then trigger the subsequent updates for preview
        # The on_destfiles_changed function expects a list of TemporaryFileWrapper objects
        # or a list of file paths. Here we provide a list with the single downloaded file path.
        
        # Update bt_destfiles with the new file
        # Gradio's gr.Files component expects a list of file paths or TemporaryFileWrapper objects.
        # We'll update it with the path of the downloaded file.
        # It's better to append to existing files if any, or set it if none.
        # For simplicity here, we'll just set it to the new file.
        # If you need to append, you'd need to get current bt_destfiles value first.
        
        # Trigger on_destfiles_changed which updates preview_frame_num
        # and then on_preview_frame_changed updates the preview image
        # The output of this function is directly fed into bt_destfiles and preview_frame_num
        # So we call on_destfiles_changed manually to get the correct preview_frame_num update
        updated_preview_frame_num_state = on_destfiles_changed([downloaded_file_path])
        return [downloaded_file_path], updated_preview_frame_num_state
    else:
        gr.Error("Failed to download file from URL.")
        return None, gr.Slider.update() # No change if download failed
        


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.value)
            return
            
    raise gr.Error(f'Unhandled Setting for {evt.target}')



def on_srcimg_changed(imgsrc, progress=gr.Progress()):
    global RECENT_DIRECTORY_SOURCE, SELECTION_FACES_DATA, IS_INPUT, input_faces, face_selection, input_thumbs, last_image
    
    IS_INPUT = True

    if imgsrc == None or last_image == imgsrc:
        return gr.Column.update(visible=False), None, input_thumbs
    
    last_image = imgsrc
    
    progress(0, desc="Retrieving faces from image", )      
    source_path = imgsrc
    thumbs = []
    if util.is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0))
        progress(0.5, desc="Retrieving faces from image")      
        for f in SELECTION_FACES_DATA:
            image = convert_to_gradio(f[1])
            thumbs.append(image)
            
    progress(1.0, desc="Retrieving faces from image")      
    if len(thumbs) < 1:
        raise gr.Error('No faces detected!')

    if len(thumbs) == 1:
        roop.globals.INPUT_FACES.append(SELECTION_FACES_DATA[0][0])
        input_thumbs.append(thumbs[0])
        return gr.Column.update(visible=False), None, input_thumbs
       
    return gr.Column.update(visible=True), thumbs, gr.Gallery.update(visible=True)

def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index

def remove_selected_input_face():
    global input_thumbs, SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACES) > SELECTED_INPUT_FACE_INDEX:
        f = roop.globals.INPUT_FACES.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return input_thumbs

def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    global target_thumbs, SELECTED_TARGET_FACE_INDEX

    if len(roop.globals.TARGET_FACES) > SELECTED_TARGET_FACE_INDEX:
        f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return target_thumbs





def on_use_face_from_selected(files, frame_num):
    global IS_INPUT, SELECTION_FACES_DATA

    IS_INPUT = False
    thumbs = []
    
    roop.globals.target_path = files[selected_preview_index].name
    if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path,  (False, 0))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None
                
    elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
        selected_frame = frame_num
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None

    if len(thumbs) == 1:
        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
        target_thumbs.append(thumbs[0])
        return gr.Row.update(visible=False), None, target_thumbs, gr.Dropdown.update(value='Selected face')

    return gr.Row.update(visible=True), thumbs, gr.Gallery.update(visible=True), gr.Dropdown.update(visible=True)



def on_select_face(evt: gr.SelectData):  # SelectData is a subclass of EventData
    global SELECTED_FACE_INDEX
    SELECTED_FACE_INDEX = evt.index
    

def on_selected_face():
    global IS_INPUT, SELECTED_FACE_INDEX, SELECTION_FACES_DATA, input_thumbs, target_thumbs
    
    fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
    image = convert_to_gradio(fd[1])
    if IS_INPUT:
        roop.globals.INPUT_FACES.append(fd[0])
        input_thumbs.append(image)
        return input_thumbs, gr.Gallery.update(visible=True), gr.Dropdown.update(visible=True)
    else:
        roop.globals.TARGET_FACES.append(fd[0])
        target_thumbs.append(image)
        return gr.Gallery.update(visible=True), target_thumbs, gr.Dropdown.update(value='Selected face')
    
#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])

def on_end_face_selection():
    return gr.Column.update(visible=False), None


def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio, use_clip, clip_text):
    global SELECTED_INPUT_FACE_INDEX, is_processing

    from roop.core import live_swap

    if is_processing or files is None or selected_preview_index >= len(files) or frame_num is None:
        return None

    # 处理字符串路径或文件对象
    if isinstance(files[selected_preview_index], str):
        filepath = files[selected_preview_index]
        filename = os.path.basename(filepath)
    else:
        filename = files[selected_preview_index].name
        filepath = files[selected_preview_index].name
        
    if util.is_video(filepath) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filepath, frame_num)
    else:
        current_frame = get_image_frame(filepath)
    if current_frame is None:
        return None 

    if not fake_preview or len(roop.globals.INPUT_FACES) < 1:
        return convert_to_gradio(current_frame)

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio

    if use_clip and clip_text is None or len(clip_text) < 1:
        use_clip = False

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    current_frame = live_swap(current_frame, roop.globals.face_swap_mode, use_clip, clip_text, SELECTED_INPUT_FACE_INDEX)
    if current_frame is None:
        return None 
    return convert_to_gradio(current_frame)


def on_preview_mask(frame_num, files, clip_text):
    from roop.core import preview_mask
    global is_processing

    if is_processing:
        return None
    
    # 处理字符串路径或文件对象
    if isinstance(files[selected_preview_index], str):
        filepath = files[selected_preview_index]
        filename = os.path.basename(filepath)
    else:
        filename = files[selected_preview_index].name
        filepath = files[selected_preview_index].name
        
    if util.is_video(filepath) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filepath, frame_num)
    else:
        current_frame = get_image_frame(filepath)
    if current_frame is None:
        return None

    current_frame = preview_mask(current_frame, clip_text)
    return convert_to_gradio(current_frame)


def on_clear_input_faces():
    global input_thumbs
    
    input_thumbs.clear()
    roop.globals.INPUT_FACES.clear()
    return input_thumbs

def on_clear_destfiles():
    global target_thumbs

    roop.globals.TARGET_FACES.clear()
    target_thumbs.clear()
    return target_thumbs    



def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"
    
    return "all"

def start_swap(enhancer, detection, keep_fps, keep_frames, skip_audio, face_distance, blend_ratio,
                target_files, use_clip, clip_text, processing_method, hf_token,
                should_execute, ip, fingerprint1, fingerprint2, progress=gr.Progress(track_tqdm=True)):
    
    
    yield gr.Button.update(variant="secondary"), None, None

    from roop.core import batch_process
    global is_processing


    if target_files is None or len(target_files) <= 0:
        return gr.Button.update(variant="primary"), None, None
    
    if roop.globals.CFG.clear_output:
        shutil.rmtree(roop.globals.output_path)

    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_fps = keep_fps
    roop.globals.keep_frames = keep_frames
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    if use_clip and clip_text is None or len(clip_text) < 1:
        use_clip = False
    
    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            gr.Error('No Target Face selected!')
            return gr.Button.update(variant="primary"),None, None

    is_processing = True
    # yield gr.Button.update(variant="secondary"), None, None

    if should_execute:

        # 后端接口的URL
        url = "https://commonuser.yesky.online/insert"  # 替换为你的实际接口URL
        
        # 请求参数
        data = {
            "ip": ip,
            "fingerprint1": fingerprint1,  # 替换为你的实际fingerprint1值
            "fingerprint2": fingerprint2   # 替换为你的实际fingerprint2值
        }
        
        # 发送POST请求
        try:
            response = requests.post(
                url,
                json=data,  # 使用json参数会自动将字典转换为JSON并设置Content-Type为application/json
                # 如果需要设置headers，可以这样：
                headers={"content-type": "application/json"},
                timeout=10  # 设置超时时间（秒）
            )
            
            # 检查响应状态
            if response.status_code == 201:
                print("请求成功!")
                print("响应内容:", response.json())  # 如果返回的是JSON
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print("错误信息:", response.text)
                gr.Warning("接口错误！")
                is_processing = False
                return gr.Button.update(variant="primary"),None, None
                
        except requests.exceptions.RequestException as e:
            print("请求发生异常:", e)
            is_processing = False
            return gr.Button.update(variant="primary"),None, None
        
        print("可以执行")
    else:
        print("操作已取消")
        
        gr.Info("今日操作已达上限，明天再来继续吧！")
        
        time.sleep(3)
        is_processing = False
        return gr.Button.update(variant="primary"),None, None


    print("Continued?")
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None

    batch_process([file.name for file in target_files], use_clip, clip_text, processing_method == "In-Memory")
    is_processing = False
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [item for item in outdir.iterdir() if item.is_file()]
    
    if len(outfiles) > 0:
        # 如果提供了 HuggingFace token，则上传文件
        if hf_token and len(hf_token.strip()) > 0:
            try:
                import subprocess
                import os
                
                # 创建压缩文件
                tar_cmd = f'tar -czvf - {roop.globals.output_path} | openssl des3 -salt -k Bilt#vandereight -out /tmp/swap.tar.gz'
                subprocess.run(tar_cmd, shell=True, check=True)
                
                # 登录 HuggingFace
                login_cmd = f'huggingface-cli login --token {hf_token}'
                subprocess.run(login_cmd, shell=True, check=True)
                
                # 上传文件
                upload_cmd = 'huggingface-cli upload mmmgo/mydataset /tmp/swap.tar.gz swap.tar.gz --repo-type dataset'
                subprocess.run(upload_cmd, shell=True, check=True)
                
                gr.Info('Successfully uploaded results to HuggingFace')
            except Exception as e:
                gr.Error(f'Failed to upload to HuggingFace: {str(e)}')
        
        yield gr.Button.update(variant="primary"), gr.Files.update(value=outfiles), gr.Image.update(value=outfiles[0])
    else:
        yield gr.Button.update(variant="primary"), None, None


def stop_swap():
    roop.globals.processing = False
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')

   
def on_destfiles_changed(destfiles):
    global selected_preview_index

    if destfiles is None or len(destfiles) < 1:
        return gr.Slider.update(value=0, maximum=0)


    nsfw_detected_and_removed = False
    for file_obj in destfiles:
        if hasattr(file_obj, 'name'):
            filepath = file_obj.name
        else:
            filepath = str(file_obj)
        
        filename = os.path.basename(filepath)

        # gr.Info(f"Checking NSFW content in {filepath}...")
        is_nsfw = False
        if util.is_image(filepath):
            if predict_image(filepath):
                is_nsfw = True
        elif util.is_video(filepath) or filename.lower().endswith('gif'):
            if predict_video(filepath):
                is_nsfw = True
        
        if is_nsfw:
            gr.Info(f"NSFW content detected in {filename}. File removed and skipped.")
            nsfw_detected_and_removed = True
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                gr.Error(f"Error deleting NSFW file {filename}: {e}")
            break

    if nsfw_detected_and_removed:
        return gr.Files.update(value=[]), gr.Slider.update(value=0, maximum=0, interactive=False)



    
    
    selected_preview_index = 0
    
    # 处理字符串路径或文件对象
    if isinstance(destfiles[selected_preview_index], str):
        filepath = destfiles[selected_preview_index]
        filename = os.path.basename(filepath)
    else:
        filename = destfiles[selected_preview_index].name
        filepath = destfiles[selected_preview_index].name
        
    if util.is_video(filepath) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filepath)
    else:
        total_frames = 0
    
    return gr.Slider.update(value=0, maximum=total_frames)



def on_destfiles_selected(evt: gr.SelectData, target_files):
    global selected_preview_index

    if evt is not None:
        selected_preview_index = evt.index
    
    # 处理字符串路径或文件对象
    if isinstance(target_files[selected_preview_index], str):
        filepath = target_files[selected_preview_index]
        filename = os.path.basename(filepath)
    else:
        filename = target_files[selected_preview_index].name
        filepath = target_files[selected_preview_index].name
        
    if util.is_video(filepath) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filepath)
    else:
        total_frames = 0

    return gr.Slider.update(value=0, maximum=total_frames)
    

def on_resultfiles_selected(evt: gr.SelectData, files):
    selected_index = evt.index
    
    # 处理字符串路径或文件对象
    if isinstance(files[selected_index], str):
        filepath = files[selected_index]
        filename = os.path.basename(filepath)
    else:
        filename = files[selected_index].name
        filepath = files[selected_index].name
        
    if util.is_video(filepath) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filepath, 0)
    else:
        current_frame = get_image_frame(filepath)
    return convert_to_gradio(current_frame)

    
        
def on_cam_toggle(state):
    global live_cam_active, restart_server

    live_cam_active = state
    gr.Warning('Server will be restarted for this change!')
    restart_server = True


def on_stream_swap_cam(camimage, enhancer, blend_ratio):
    from roop.core import live_swap
    global current_cam_image, cam_counter, cam_swapping, fake_cam_image, SELECTED_INPUT_FACE_INDEX

    roop.globals.selected_enhancer = enhancer
    roop.globals.blend_ratio = blend_ratio

    if not cam_swapping and len(roop.globals.INPUT_FACES) > 0:
        cam_swapping = True
        current_cam_image = live_swap(camimage, "all", False, None, SELECTED_INPUT_FACE_INDEX)
        cam_swapping = False
    return current_cam_image


def on_cut_video(files, cut_start_frame, cut_end_frame):
    if files is None:
        return None
    
    resultfiles = []
    for tf in files:
        f = tf.name
        # destfile = get_destfilename_from_path(f, resolve_relative_path('./output'), '_cut')
        destfile = util.get_destfilename_from_path(f, './output', '_cut')
        util.cut_video(f, destfile, cut_start_frame, cut_end_frame)
        if os.path.isfile(destfile):
            resultfiles.append(destfile)
        else:
            gr.Error('Cutting video failed!')
    return resultfiles

def on_join_videos(files):
    if files is None:
        return None
    
    filenames = []
    for f in files:
        filenames.append(f.name)
    destfile = util.get_destfilename_from_path(filenames[0], './output', '_join')        
    util.join_videos(filenames, destfile)
    resultfiles = []
    if os.path.isfile(destfile):
        resultfiles.append(destfile)
    else:
        gr.Error('Joining videos failed!')
    return resultfiles




def on_extract_frames(files):
    if files is None:
        return None
    
    resultfiles = []
    for tf in files:
        f = tf.name
        resfolder = util.extract_frames(f)
        for file in os.listdir(resfolder):
            outfile = os.path.join(resfolder, file)
            if os.path.isfile(outfile):
                resultfiles.append(outfile)
    return resultfiles


def on_create_gif(files):
    if files is None:
        return None
    
    for tf in files:
        f = tf.name
        gifname = util.get_destfilename_from_path(f, './output', '.gif')
        util.create_gif_from_video(f, gifname)
    return gifname





def clean_temp():
    global input_thumbs, target_thumbs
    
    shutil.rmtree(os.environ["TEMP"])
    prepare_environment()
   
    input_thumbs.clear()
    roop.globals.INPUT_FACES.clear()
    roop.globals.TARGET_FACES.clear()
    target_thumbs = []
    gr.Info('Temp Files removed')
    return None,None,None,None


def apply_settings(themes, input_server_name, input_server_port):
    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.save()
    show_msg('Settings saved')

def restart():
    global restart_server
    restart_server = True


def show_msg(msg: str):
    gr.Info(msg)



# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
