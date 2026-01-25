import math

class HolospaceProjection:
    def __init__(self):
        self.vertices = []
        self.camera_pos = [0, 0, -5]
        self.angle_x = 0.5  # Initial pitch
        self.angle_y = 0.5  # Initial yaw
        self.angle_z = 0.0
        
        self.fov = 500  # Field of View / Zoom scale
        
    def rotate_point(self, x, y, z):
        """Applies 3D rotation matrices."""
        # Rotation X
        cos_x = math.cos(self.angle_x)
        sin_x = math.sin(self.angle_x)
        
        y1 = y * cos_x - z * sin_x
        z1 = z * cos_x + y * sin_x
        x1 = x
        
        # Rotation Y
        cos_y = math.cos(self.angle_y)
        sin_y = math.sin(self.angle_y)
        
        x2 = x1 * cos_y + z1 * sin_y
        z2 = z1 * cos_y - x1 * sin_y
        y2 = y1
        
        return x2, y2, z2 # We ignore Z rotation for simple orbiting

    def project(self, points, screen_width, screen_height):
        """Projects a list of (x,y,z, color) tuples to 2D screen coordinates."""
        projected_points = []
        
        center_x = screen_width / 2
        center_y = screen_height / 2
        
        for x, y, z, color in points:
            # 1. Rotate
            rx, ry, rz = self.rotate_point(x, y, z)
            
            # 2. Camera Transform (Simple translation for now)
            # Viewer is at (0,0, -viewer_distance), looking at origin
            # Depth check
            depth = rz - self.camera_pos[2]
            
            if depth <= 0.1: continue # Clip points behind camera
            
            # 3. Projection (Weak Perspective)
            scale = self.fov / depth
            
            screen_x = (rx * scale) + center_x
            screen_y = (ry * scale) + center_y # Y is typically inverted in 3D but DPG 0,0 is top-left
            
            # Size attenuation
            size = max(2, 100 / depth)
            
            projected_points.append({
                'x': screen_x,
                'y': screen_y,
                'size': size,
                'color': color,
                'depth': depth # For Z-sorting if needed
            })
            
        # Z-Sort (Painter's Algorithm) - draw furthest first
        projected_points.sort(key=lambda p: p['depth'], reverse=True)
        
        return projected_points

    def set_rotation(self, dx, dy):
        """Update rotation angles based on mouse delta."""
        self.angle_y += dx * 0.01
        self.angle_x += dy * 0.01
