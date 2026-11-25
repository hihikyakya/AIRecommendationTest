from pathlib import Path
import pandas as pd


def parse_table(filepath: str | Path | None = None) -> pd.DataFrame:
    """MovieLens 1M evaluation dataset을 불러옵니다.
    
    기본 경로는 프로젝트 루트의 resources/dataset/movielens_1m_eval.csv 입니다.
    """
    if filepath is None:
        # 현재 파일 위치: src/ai_rec_test/services/parse_data.py
        current_dir = Path(__file__).resolve().parent
        # 프로젝트 루트 (src의 상위)
        project_root = current_dir.parent.parent.parent
        filepath = project_root / "resources" / "dataset" / "movielens_1m_eval.csv"
        
        # 만약 해당 경로에 없다면 현재 작업 디렉토리 기준도 탐색
        if not filepath.exists():
            cwd_path = Path.cwd() / "resources" / "dataset" / "movielens_1m_eval.csv"
            if cwd_path.exists():
                filepath = cwd_path

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {filepath}")

    df = pd.read_csv(filepath)
    return df
