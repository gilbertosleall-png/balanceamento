"""
balance_engine.py

Funções públicas:
- balance_line(df_tasks, num_ops, cycle_time=None, solver_time=20, use_heuristic=True)
    Recebe dataframe com colunas mínimas: TaskID, Time, TaskName (opcional), Predecessors (opcional)
    Retorna: (df_assignment, kpis_dict, df_gantt)

- generate_gantt(df_gantt)
    Recebe df com colunas: Station, Task, Start, Finish, Duration -> retorna plotly figure

- export_excel_report(df_assign, df_gantt)
    Retorna bytes prontos para download (xlsx)

- export_pdf_report(df_assign)
    Retorna bytes com PDF simples contendo tabela de atribuições

Dependências: pandas, ortools, plotly, openpyxl, matplotlib
"""

import io
import pandas as pd
from ortools.sat.python import cp_model
import plotly.express as px
import matplotlib.pyplot as plt


def balance_line(df_tasks: pd.DataFrame, num_ops: int, cycle_time: float = None, solver_time: int = 20, use_heuristic: bool = True):
    """
    Balanceia as tarefas entre `num_ops` estações.

    df_tasks deve conter colunas: TaskID, Time. Opcional: TaskName, Predecessors.
    """
    # Validate
    if 'TaskID' not in df_tasks.columns or 'Time' not in df_tasks.columns:
        raise ValueError('df_tasks precisa ter colunas TaskID e Time')

    df = df_tasks.copy()
    if 'TaskName' not in df.columns:
        df['TaskName'] = df['TaskID']
    if 'Predecessors' not in df.columns:
        df['Predecessors'] = ''

    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    if df['Time'].isnull().any():
        raise ValueError('Existem valores inválidos na coluna Time')

    total_time = float(df['Time'].sum())
    if cycle_time is None:
        cycle_time = total_time / max(1, num_ops)

    # build precedences
    id_set = set(df['TaskID'])
    precedences = []
    for _, r in df.iterrows():
        preds = str(r['Predecessors']).strip()
        if preds:
            for p in [p.strip() for p in preds.split(',') if p.strip()]:
                if p not in id_set:
                    raise ValueError(f'Predecessor {p} não existe em TaskID')
                precedences.append((p, r['TaskID']))

    # Heuristic fallback for large instances
    TASK_COUNT_THRESHOLD = 200
    if use_heuristic and len(df) > TASK_COUNT_THRESHOLD:
        # Simple Largest Candidate Rule: sort by time desc and assign to station with min load
        rows = df.sort_values('Time', ascending=False).to_dict('records')
        station_loads = {i+1: 0 for i in range(num_ops)}
        assignment = []
        for r in rows:
            # choose station with min load that doesn't exceed cycle_time if possible
            possible = sorted(station_loads.items(), key=lambda x: x[1])
            placed = False
            for s, load in possible:
                if load + int(r['Time']) <= cycle_time:
                    station_loads[s] += int(r['Time'])
                    assignment.append({'TaskID': r['TaskID'], 'TaskName': r['TaskName'], 'Time': int(r['Time']), 'Station': s})
                    placed = True
                    break
            if not placed:
                # place in station with min load regardless
                smin = min(station_loads, key=lambda k: station_loads[k])
                station_loads[smin] += int(r['Time'])
                assignment.append({'TaskID': r['TaskID'], 'TaskName': r['TaskName'], 'Time': int(r['Time']), 'Station': smin})
        df_assign = pd.DataFrame(assignment).sort_values('Station')
    else:
        # CP-SAT model
        model = cp_model.CpModel()
        TIDs = list(df['TaskID'])
        station_vars = {tid: model.NewIntVar(1, num_ops, f'st_{tid}') for tid in TIDs}
        y = {}
        for tid in TIDs:
            for s in range(1, num_ops+1):
                y[(tid, s)] = model.NewBoolVar(f'y_{tid}_{s}')
            model.Add(sum(y[(tid, s)] for s in range(1, num_ops+1)) == 1)
            model.Add(sum(y[(tid, s)] * s for s in range(1, num_ops+1)) == station_vars[tid])

        # load constraints
        for s in range(1, num_ops+1):
            model.Add(sum(int(df.loc[df['TaskID'] == tid, 'Time'].values[0]) * y[(tid, s)] for tid in TIDs) <= int(round(cycle_time)))

        # precedence
        for pre, suc in precedences:
            model.Add(station_vars[pre] <= station_vars[suc])

        # objective minimize total idle
        station_loads = {s: model.NewIntVar(0, int(total_time), f'load_{s}') for s in range(1, num_ops+1)}
        for s in station_loads:
            model.Add(station_loads[s] == sum(int(df.loc[df['TaskID'] == tid, 'Time'].values[0]) * y[(tid, s)] for tid in TIDs))
        total_load = model.NewIntVar(0, int(total_time), 'total_load')
        model.Add(total_load == sum(station_loads[s] for s in station_loads))
        cycle_int = int(round(cycle_time))
        total_idle = model.NewIntVar(0, num_ops * cycle_int, 'total_idle')
        model.Add(total_idle == num_ops * cycle_int - total_load)
        model.Minimize(total_idle)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(solver_time)
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        assignment = []
        for tid in TIDs:
            s = solver.Value(station_vars[tid])
            ttime = int(df.loc[df['TaskID'] == tid, 'Time'].values[0])
            assignment.append({'TaskID': tid, 'TaskName': df.loc[df['TaskID'] == tid, 'TaskName'].values[0], 'Time': ttime, 'Station': s})
        df_assign = pd.DataFrame(assignment).sort_values('Station')

    # KPIs and gantt
    loads = df_assign.groupby('Station')['Time'].sum().reindex(range(1, num_ops+1), fill_value=0)
    utilization = total_time / (num_ops * cycle_time) if cycle_time > 0 else 0
    total_idle_val = (num_ops * cycle_time) - total_time

    kpis = {
        'cycle_time': float(cycle_time),
        'total_task_time': float(total_time),
        'utilization': float(utilization),
        'total_idle': float(total_idle_val)
    }

    # build gantt df
    gantt_rows = []
    for s in range(1, num_ops+1):
        tasks_s = df_assign[df_assign['Station'] == s]
        start = 0
        for _, r in tasks_s.iterrows():
            gantt_rows.append({'Station': f'Op {s}', 'Task': f"{r['TaskID']} - {r['TaskName']}", 'Start': start, 'Finish': start + int(r['Time']), 'Duration': int(r['Time'])})
            start += int(r['Time'])
    df_gantt = pd.DataFrame(gantt_rows)

    return df_assign.reset_index(drop=True), kpis, df_gantt


def generate_gantt(df_gantt: pd.DataFrame):
    if df_gantt.empty:
        return None
    fig = px.timeline(df_gantt, x_start='Start', x_end='Finish', y='Station', color='Station', hover_data=['Task', 'Duration'])
    fig.update_yaxes(autorange='reversed')
    return fig


def export_excel_report(df_assign: pd.DataFrame, df_gantt: pd.DataFrame):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_assign.to_excel(writer, index=False, sheet_name='assignment')
        df_gantt.to_excel(writer, index=False, sheet_name='gantt')
    buf.seek(0)
    return buf.getvalue()


def export_pdf_report(df_assign: pd.DataFrame):
    fig_pdf, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    if not df_assign.empty:
        table_df = df_assign[['TaskID', 'TaskName', 'Time', 'Station']].copy()
        table_df.columns = ['TaskID', 'TaskName', 'Time(s)', 'Station']
        ax.table(cellText=table_df.values, colLabels=table_df.columns, loc='center', cellLoc='left')
    plt.tight_layout()
    buf = io.BytesIO()
    fig_pdf.savefig(buf, format='pdf')
    plt.close(fig_pdf)
    buf.seek(0)
    return buf.getvalue()
