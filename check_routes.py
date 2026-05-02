import main

print("Registered routes:")
for route in main.app.routes:
    if hasattr(route, 'path') and '/cameras' in route.path:
        print(f"  {route.methods if hasattr(route, 'methods') else 'N/A'} {route.path}")
