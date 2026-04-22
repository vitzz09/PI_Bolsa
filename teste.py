import customtkinter as ctk
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------
# Função para buscar API
# ---------------------------
def carregar_dados():
    url = "https://apisidra.ibge.gov.br/DescritoresTabela/t/"
    response = requests.get(url)
    dados = response.json()

    nomes = [user["name"] for user in dados]
    ids = [user["id"] for user in dados]

    df = pd.DataFrame({
        "nome": nomes,
        "id": ids
    })

    return df

# ---------------------------
# Função para criar gráfico
# ---------------------------
def gerar_grafico():
    df = carregar_dados()

    fig, ax = plt.subplots()
    ax.bar(df["nome"], df["id"])
    ax.set_title("Dados da API")

    # limpar gráfico antigo
    for widget in frame_grafico.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas.get_tk_widget().pack()

# ---------------------------
# Interface
# ---------------------------
ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.geometry("900x700")
app.title("API + Gráfico")

titulo = ctk.CTkLabel(app, text="Consumindo API", font=("Arial", 20))
titulo.pack(pady=10)

botao = ctk.CTkButton(app, text="Carregar Dados", command=gerar_grafico)
botao.pack(pady=10)

frame_grafico = ctk.CTkFrame(app)
frame_grafico.pack(fill="both", expand=True, padx=20, pady=20)

app.mainloop()