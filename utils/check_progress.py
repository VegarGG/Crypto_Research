from arcticdb import Arctic
try:
    from config import ARCTIC_URI
except ImportError:
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent
    arctic_store = project_root / "arctic_store"
    ARCTIC_URI = f"lmdb://{arctic_store}"

def check_progress():
    print(f"Connecting to {ARCTIC_URI}")
    arctic = Arctic(ARCTIC_URI)
    libs = arctic.list_libraries()
    print(f"Libraries found: {libs}")
    
    for lib in libs:
        try:
            l = arctic[lib]
            if 'BTCUSDT' in l.list_symbols():
                info = l.read_metadata('BTCUSDT')
                # ArcticDB metadata might not show write time easily in all versions, 
                # but existence of symbol means write happened.
                data = l.read('BTCUSDT').data
                print(f"  - {lib}: BTCUSDT present, {len(data)} rows")
            else:
                print(f"  - {lib}: BTCUSDT NOT found")
        except Exception as e:
            print(f"  - {lib}: Error reading ({e})")

if __name__ == "__main__":
    check_progress()
