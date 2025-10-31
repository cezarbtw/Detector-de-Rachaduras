from ultralytics import YOLO


def main():
    print("Iniciando o processo de treinamento...")

    model = YOLO('yolov8s.pt')

    try:
        results = model.train(
            data='data.yaml',
            epochs=100,
            imgsz=640,
            batch=64,
            device=0,
            workers=2,
            name='yolov8l_DETECCAOSMALL_rachaduras_v1'
        )
        print("Treinamento de SEGMENTAÇÃO concluído com sucesso!")
        print("Iniciando avaliação no conjunto de TESTE...")
        metrics = model.val(split='test')
        print("Métricas de teste:")
        print(metrics)

    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")


if __name__ == '__main__' :main()