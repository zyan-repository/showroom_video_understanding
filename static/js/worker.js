let updateInterval = null;  // Store setInterval ID for clearing it when needed
let prev_camera_name = null;  // Store previous camera name

self.onmessage = function(e) {
    var port = e.data.port; // 获取端口号
    var ip = e.data.ip; // 获取 IP 地址
    var camera_name = e.data.camera_name; // Get camera name
    var ip_port = ip + ":" + port;
    // Only clear the interval and start a new one if the camera name has changed
    if (camera_name !== prev_camera_name) {
        // If there is an update loop running, clear it
        if (updateInterval !== null) {
            clearInterval(updateInterval);
        }
        function updateInferenceResult() {
            fetch(`http://${ip_port}/inference/${camera_name}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('HTTP error, status = ' + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    let html = data.result.replace(/\n/g, "<br>");
                    postMessage({type: 'inference', data: html});
                    // Note that data.category is expected to be a number from 1 to 8
                    let category = data.category;
                    postMessage({type: 'category', data: category});
                })
                .catch(error => {
                    console.log(error.message);
                });
        }

        // Update inference result every 0.03 seconds
        updateInterval = setInterval(updateInferenceResult, 30);
        
        // Update the previous camera name
        prev_camera_name = camera_name;
    }
};
