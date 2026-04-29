#  Sistema de Detecção e Segmentação de Rachaduras com YOLOv8 e PyQt5

##  Descrição Geral
Este projeto implementa um sistema para **detecção e segmentação de rachaduras em imagens**, utilizando o modelo **YOLOv8** da biblioteca **Ultralytics**.  
O sistema inclui uma interface gráfica desenvolvida com **PyQt5**, que permite ao usuário selecionar imagens, processá-las e visualizar os resultados da detecção.

---

## ⚙️ Estrutura do Projeto

```
📦 detector-rachaduras-yolov8
 ┣ 📂 imagens/                # Pasta de imagens para teste
 ┣ 📂 runs/                   # Resultados do treinamento e inferência
 ┣ 📜 train_model.py          # Script de treinamento do YOLOv8
 ┣ 📜 app_interface.py        # Interface gráfica em PyQt5
 ┣ 📜 requirements.txt        # Dependências do projeto
 ┗ 📜 README.md               # Documentação do projeto
```

---

##  Funcionalidades Principais

✅ Treinamento personalizado de modelo YOLOv8  
✅ Detecção e segmentação de rachaduras em imagens  
✅ Interface gráfica intuitiva com PyQt5  
✅ Visualização dos resultados diretamente na aplicação  
✅ Suporte a carregamento de novas imagens  

---

##  Tecnologias Utilizadas

| Categoria | Ferramenta |
|------------|-------------|
| Linguagem | Python 3.10+ |
| IA/Visão Computacional | YOLOv8 (Ultralytics) |
| Interface Gráfica | PyQt5 |
| Processamento de Imagem | OpenCV |
| Ambiente Virtual | venv ou conda |

---

## 🖥️ Interface Gráfica (PyQt5)

A interface permite:  
- Escolher uma imagem do computador  
- Executar a detecção com YOLOv8  
- Visualizar o resultado processado  

```python
QPushButton("Selecionar Imagem")
QPushButton("Detectar Rachaduras")
QLabel("Resultado da Detecção")
```

---

## 📚 Treinamento do Modelo

O treinamento é feito com o script `train_model.py`, utilizando a estrutura padrão do YOLOv8.

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='data.yaml', epochs=100, imgsz=640, batch=64, device=0)
```

O arquivo `data.yaml` define o caminho das imagens de treino e validação.

---

## ▶️ Execução da Interface

Para rodar a aplicação gráfica:

```bash
python app_interface.py
```

---

## 🧑‍💻 Autoria

Desenvolvido por:

- [**Abraão Cezar**](https://github.com/cezarbtw)  
- [**Gabriel Oliveira**](https://github.com/GabrielOlNascimento)

> Projeto desenvolvido como parte da APS do curso de Ciência da Computação, integrando Inteligência Artificial e Visão Computacional com interface gráfica em PyQt5.

---

##  Dependências

Crie e ative um ambiente virtual, depois instale as dependências com:

```bash
pip install -r requirements.txt
```

Exemplo de `requirements.txt`:

```
ultralytics
pyqt5
opencv-python
numpy
```

---

##  Resultados

Os resultados de detecção são armazenados automaticamente em:

```
runs/detect/predict
```

---

##  Observações Finais

- O modelo YOLOv8 deve ser ajustado de acordo com o dataset utilizado.  
- Certifique-se de manter as pastas `train/` e `valid/` corretamente configuradas no `data.yaml`.  
- A interface pode ser expandida para incluir novas funcionalidades, como gravação de logs ou integração com banco de dados.

---

##  Licença

Este projeto é de uso acadêmico e pode ser adaptado para fins de pesquisa e aprendizado.

---

###  Repositório Oficial

🔗 [GitHub - detector-rachaduras-yolov8](https://github.com/cezarbtw/detector-rachaduras-yolov8)
