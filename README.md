# 🤖 `vlm_robot_agent`

### Author: Edison Bejarano


Library designed to use VLMs in the context of robotics actions for planning and interactions



Un **agente robótico inteligente** basado en modelos de lenguaje visual (VLM), que puede percibir el entorno desde una imagen, planificar acciones y decidir entre **navegar** o **interactuar con personas** para alcanzar un objetivo dado (como entrar en una habitación, buscar un baño, etc).

---

## 🚀 Características

- 📷 **Percepción visual** usando un VLM de OpenAI
- 🧠 **Razonamiento basado en objetivos** (macro y micro-goals)
- 🧭 Acciones de **navegación**:
  - `forward`, `left`, `right`, `forward_left`, `forward_right`
- 🙋 Acciones de **interacción**:
  - Conversar con una persona que bloquea el paso
  - Hacer gestos para pedir que se mueva
- 💾 **Memoria de interacciones** y lectura/ejecución de prompts desde un folder

---

## 📦 Instalación

```bash
pip install vlm_robot_agent
```

---

## 🛠 Uso básico

```python
from vlm_robot_agent import VLMRobotAgent

agent = VLMRobotAgent(prompt_folder="./prompts")

image = obtener_imagen_de_tu_robot()
goal = "entrar a la oficina 3"

# Loop de ejecución
while True:
    action = agent.step(image, goal)
    ejecutar_action_en_robot(action)
    if objetivo_cumplido():
        break
```

---

## 📁 Estructura de prompts
Los prompts se almacenan como archivos `.json` dentro del folder configurado, y puedes cargarlos con:

```python
prompts = agent.load_prompts()
```

---

## 🧩 Integración con robots
- Puede usarse en sistemas ROS, simuladores como Gazebo, o cualquier entorno de robots.
- El agente necesita:
  - Imagen actual del entorno (`image`)
  - Objetivo a cumplir (`goal`)
  - Una función que ejecute la acción devuelta (`Navigate`, `Interact`)

---

## 📚 Ejemplo de acciones
```python
from vlm_robot_agent import Navigate, Interact

# Navegar hacia adelante
Navigate(direction="forward")

# Pedir a una persona que se mueva
Interact(strategy="ask_to_move")
```

---

## 📄 Licencia
MIT

---

## 🧠 Futuras mejoras
- Seguimiento de progreso con `StateTracker`
- Manejo de múltiples agentes o flujos conversacionales
- Soporte para entrada multimodal (texto + imagen)


<p align="center">
  <img src="img/edison-bejarano.png" alt="Edison Bejarano" width="300"/>
</p>
