import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

CAMINHO_MODELO_GRANDE = r'C:\Users\bifed\Desktop\APS-V2-Revisao.v1i.yolov8\runs\detect\yolov8l_DETECCAOSMALL_rachaduras_v14\weights\best.pt'
CAMINHO_MODELO_MEDIO = r'C:\Users\bifed\Desktop\APS-V2-Revisao.v1i.yolov8\runs\detect\yolov8l_DETECCAOSMALL_rachaduras_v14\weights\best.pt'
DIRETORIO_SAIDA = 'inference_results'
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

FOLHA_ESTILO = """
QWidget {
    background-color: #2B2B2B;
    color: #F0F0F0;
    font-family: "Segoe UI", Arial, sans-serif;
}
QLabel {
    font-size: 14pt;
    padding: 10px;
}
QLabel#ImageResultLabel {
     min-height: 300px;
     min-width: 400px;
     background-color: #1E1E1E;
     border: 1px solid #3C3C3C;
     border-radius: 8px;
}
QPushButton {
    background-color: #007ACC;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 15px;
    font-size: 11pt;
}
QPushButton:hover {
    background-color: #008DD0;
}
QPushButton:pressed {
    background-color: #005A8C;
}
QPushButton#ROIButton {
    background-color: #5cb85c;
    margin-top: 5px;
}
QPushButton#ROIButton:hover {
    background-color: #4cae4c;
}
QPushButton#ROIButton:pressed {
    background-color: #398439;
}
"""


def executar_segmentacao_tempo_real(caminho_modelo):
    nome_janela = "Analise em Tempo Real (Pressione 'e' para sair)"
    print(f"Carregando modelo de tempo real: {caminho_modelo}")

    try:
        modelo = YOLO(caminho_modelo)
        captura = cv2.VideoCapture(0)

        if not captura.isOpened():
            print("Erro: Não foi possível abrir a webcam.")
            QMessageBox.critical(None, "Erro Câmera",
                                 "Não foi possível abrir a webcam.")
            return

        print("Tentando definir a resolução da câmera para 1280x720...")
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        largura_real = captura.get(cv2.CAP_PROP_FRAME_WIDTH)
        altura_real = captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolução real obtida: {largura_real}x{altura_real}")

        cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            nome_janela, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        print("Iniciando câmera... Pressione 'e' na janela do vídeo para sair.")
        while True:
            ret, quadro = captura.read()
            if not ret:
                break

            resultados = modelo(quadro, device='cpu', verbose=False, conf=0.7)
            quadro_anotado = resultados[0].plot()

            cv2.imshow(nome_janela, quadro_anotado)

            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    except Exception as e:
        print(f"Ocorreu um erro na execução em tempo real: {e}")
        if not os.path.exists(caminho_modelo):
            print("ERRO: O caminho do modelo não foi encontrado...")
    finally:
        if 'captura' in locals() and captura.isOpened():
            captura.release()
        cv2.destroyAllWindows()
        print("Câmera encerrada.")


class JanelaPrincipal(QWidget):

    def __init__(self):
        super().__init__()
        self.janela_resultado = None
        self.pixmap_atual = None
        self.iniciar_ui()
        self.setStyleSheet(FOLHA_ESTILO)

    def iniciar_ui(self):
        self.setWindowTitle("Detector de Rachaduras (APS)")
        self.resize(550, 400)

        layout_principal = QVBoxLayout(self)
        layout_principal.setContentsMargins(20, 20, 20, 20)

        rotulo = QLabel("Escolha o modo de operação:")
        rotulo.setAlignment(Qt.AlignCenter)
        layout_principal.addWidget(rotulo)

        btn_processar_imagem = QPushButton("Processar Imagem Inteira")
        btn_processar_imagem.clicked.connect(
            lambda: self.executar_segmentacao_upload(CAMINHO_MODELO_GRANDE)
        )
        layout_principal.addWidget(btn_processar_imagem)

        btn_processar_roi = QPushButton("Selecionar e Processar Área (ROI)")
        btn_processar_roi.setObjectName("ROIButton")
        btn_processar_roi.clicked.connect(
            lambda: self.executar_segmentacao_roi_upload(CAMINHO_MODELO_GRANDE)
        )
        layout_principal.addWidget(btn_processar_roi)

        btn_iniciar_camera = QPushButton("Iniciar Câmera")
        btn_iniciar_camera.clicked.connect(
            lambda: executar_segmentacao_tempo_real(CAMINHO_MODELO_MEDIO)
        )
        layout_principal.addWidget(btn_iniciar_camera)

        layout_principal.setSpacing(15)

    def executar_segmentacao_upload(self, caminho_modelo):
        print(f"Carregando modelo: {caminho_modelo}")
        caminho_arquivo, _ = QFileDialog.getOpenFileName(
            self, "Selecione a imagem INTEIRA", "", "Imagens (*.jpg *.jpeg *.png *.bmp);;Todos os arquivos (*.*)")

        if not caminho_arquivo:
            return

        try:
            modelo = YOLO(caminho_modelo)
            print(f"Processando imagem inteira: {caminho_arquivo}")

            resultados = modelo(caminho_arquivo, device='cpu', conf=0.5)
            imagem_anotada = resultados[0].plot()

            nome_base = os.path.basename(caminho_arquivo)
            caminho_saida = os.path.join(
                DIRETORIO_SAIDA, f"result_{nome_base}")
            cv2.imwrite(caminho_saida, imagem_anotada)

            print(f"Imagem salva: {caminho_saida}")
            self.mostrar_imagem_na_janela(
                imagem_anotada, "Resultado - Imagem Inteira")

        except Exception as e:
            print(f"Erro ao processar imagem inteira: {e}")
            if not os.path.exists(caminho_modelo):
                print("ERRO: Modelo não encontrado...")
            QMessageBox.critical(self, "Erro Processamento", str(e))

    def executar_segmentacao_roi_upload(self, caminho_modelo):
        print(f"Carregando modelo: {caminho_modelo}")
        caminho_arquivo, _ = QFileDialog.getOpenFileName(
            self, "Selecione a imagem para escolher a área", "", "Imagens (*.jpg *.jpeg *.png *.bmp);;Todos os arquivos (*.*)")

        if not caminho_arquivo:
            return

        try:
            imagem = cv2.imread(caminho_arquivo)
            if imagem is None:
                QMessageBox.warning(
                    self, "Erro", "Não foi possível carregar a imagem.")
                return

            cv2.destroyAllWindows()
            roi = cv2.selectROI("Selecione a Area (Pressione Enter)",
                                imagem, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecione a Area (Pressione Enter)")

            # roi = (x, y, w, h)
            if roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                imagem_cortada = imagem[y:y+h, x:x+w]
                print(f"Área selecionada: x={x}, y={y}, w={w}, h={h}")

                modelo = YOLO(caminho_modelo)
                print(f"Processando área selecionada...")
                resultados = modelo(imagem_cortada, device='cpu', conf=0.5)

                if not resultados or len(resultados[0].boxes) == 0:
                    corte_anotado = imagem_cortada
                    print("Nenhuma rachadura detectada na área selecionada.")
                    QMessageBox.information(
                        self, "Resultado ROI", "Nenhuma rachadura detectada na área selecionada.")
                else:
                    corte_anotado = resultados[0].plot()

                nome_base = os.path.basename(caminho_arquivo)
                caminho_saida = os.path.join(
                    DIRETORIO_SAIDA, f"result_roi_{nome_base}")
                cv2.imwrite(caminho_saida, corte_anotado)

                print(f"Área processada salva: {caminho_saida}")
                self.mostrar_imagem_na_janela(
                    corte_anotado, "Resultado - Área Selecionada")

            else:
                print("Seleção de área cancelada ou inválida.")

        except Exception as e:
            print(f"Erro ao processar área selecionada: {e}")
            if not os.path.exists(caminho_modelo):
                print("ERRO: Modelo não encontrado...")
            QMessageBox.critical(self, "Erro Processamento ROI", str(e))
        finally:
            cv2.destroyAllWindows()

    def mostrar_imagem_na_janela(self, array_imagem_bgr, titulo_janela="Resultado"):

        imagem_rgb = cv2.cvtColor(array_imagem_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = imagem_rgb.shape
        bytes_por_linha = ch * w
        imagem_qt = QImage(imagem_rgb.data, w, h,
                           bytes_por_linha, QImage.Format_RGB888)
        self.pixmap_atual = QPixmap.fromImage(imagem_qt)

        if self.janela_resultado and self.janela_resultado.isVisible():
            self.janela_resultado.close()

        self.janela_resultado = QWidget()
        self.janela_resultado.setWindowTitle(titulo_janela)

        layout = QVBoxLayout()
        self.rotulo_resultado = QLabel()
        self.rotulo_resultado.setObjectName("ImageResultLabel")
        self.rotulo_resultado.setAlignment(Qt.AlignCenter)
        self.rotulo_resultado.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.rotulo_resultado)

        self.janela_resultado.setLayout(layout)

        largura_inicial = min(self.pixmap_atual.width(), 800)
        altura_inicial = min(self.pixmap_atual.height(), 600)
        self.janela_resultado.resize(largura_inicial, altura_inicial)

        self.janela_resultado.resizeEvent = self.janela_resultado_redimensionada

        self.escalonar_imagem_resultado()
        self.janela_resultado.show()

    def escalonar_imagem_resultado(self):
        if self.janela_resultado and self.pixmap_atual:
            pixmap_escalonado = self.pixmap_atual.scaled(
                self.rotulo_resultado.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.rotulo_resultado.setPixmap(pixmap_escalonado)

    def janela_resultado_redimensionada(self, evento):
        self.escalonar_imagem_resultado()


if __name__ == "__main__":
    if not os.path.exists(CAMINHO_MODELO_GRANDE) or not os.path.exists(CAMINHO_MODELO_MEDIO):
        print("ERRO: Caminhos dos modelos não encontrados.")
        print("Verifique os caminhos absolutos para CAMINHO_MODELO_GRANDE e CAMINHO_MODELO_MEDIO.")
        try:
            app_erro = QApplication(sys.argv)
            QMessageBox.critical(None, "Erro de Configuração",
                                 "Caminho do(s) modelo(s) não encontrado. Verifique os caminhos no script.")
        except:
            pass
    else:
        app = QApplication(sys.argv)
        janela = JanelaPrincipal()
        janela.show()
        sys.exit(app.exec_())
