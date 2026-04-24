import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

def generate_plot(plot_name, model, dataset, selected_inputs, selected_output):
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    X = dataset[selected_inputs]
    y = dataset[selected_output] if selected_output else None

    if plot_name == "Scatter Plot":
        if selected_output:
            ax.scatter(X.iloc[:, 0], y)
            ax.set_xlabel(selected_inputs[0])
            ax.set_ylabel(selected_output)
        else:
            sns.pairplot(dataset[selected_inputs])
    elif plot_name == "Confusion Matrix":
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    else:
        raise ValueError("Unsupported plot type")
    
    return fig