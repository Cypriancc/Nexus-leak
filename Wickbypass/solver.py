from flask import Flask, request, jsonify
import requests
from PIL import Image
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/wick.pt')

class wick:
    def __init__(self, url):
        self.url = url
        self.color = "32CF7E"

    def process(self, img, hex_color, tolerance=20):
        image_data = img.load()
        height, width = img.size
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r_min, r_max = max(0, r - tolerance), min(255, r + tolerance)
        g_min, g_max = max(0, g - tolerance), min(255, g + tolerance)
        b_min, b_max = max(0, b - tolerance), min(255, b + tolerance)
        for loop1 in range(height):
            for loop2 in range(width):
                try:
                    pixel_r, pixel_g, pixel_b, _ = image_data[loop1, loop2]
                except ValueError:
                    pixel_r, pixel_g, pixel_b = image_data[loop1, loop2]
                if not (r_min <= pixel_r <= r_max and g_min <= pixel_g <= g_max and b_min <= pixel_b <= b_max):
                    image_data[loop1, loop2] = 0, 0, 0, 0
        return img    
    
    def solve_captcha(self):
        try:
            img = Image.open(requests.get(self.url, stream=True).raw)
            if self.color is not None:
              img = self.process(img, self.color)   
            result = model(img)     
            a = result.pandas().xyxy[0].sort_values('xmin')
            while len(a) > 6:
                lines = a.confidence
                linev = min(a.confidence)
                for line in lines.keys():
                    if lines[line] == linev:
                        a = a.drop(line)      
            result = ""
            for _, key in a.name.items():
                result = result + key     
            return result
        except:
            return None

app = Flask(__name__)

@app.route('/api/captcha/solve', methods=['POST'])
def solve_api():
    data = request.get_json()
    captcha = data.get("captcha")
    solver = wick(captcha)
    result = solver.solve_captcha()

    if result is None or len(result) < 3:
        print("Failed to solve wick captcha")
        return jsonify({"success": False, "message": "SOLVE_FAILED"})
    
    print(f"Solved Wick Captcha : {result}")
    return jsonify({"success": True, "solve": result})

if __name__ == "__main__":
    app.run(port=3007)