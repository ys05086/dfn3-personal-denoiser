"""Known virtual cable pairs. Never silently route a microphone to speakers."""
import re


def receiver_name(output_name):
    name = output_name.split("(")[0].strip()
    if re.fullmatch(r"CABLE(?:-[A-D])? Input", name, re.IGNORECASE):
        return re.sub("Input$", "Output", name, flags=re.IGNORECASE)
    if re.fullmatch(r"VoiceMeeter(?: Aux| VAIO3)? Input", name, re.IGNORECASE):
        return re.sub("Input$", "Output", name, flags=re.IGNORECASE)
    return None


def virtual_outputs(devices, hosts):
    choices = []
    priority = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}
    for index, device in enumerate(devices):
        host = hosts[device["hostapi"]]["name"]
        receiver = receiver_name(device["name"])
        if device["max_output_channels"] and receiver and host in priority:
            choices.append((index, device["name"], host, receiver))
    return sorted(choices, key=lambda choice: (priority[choice[2]], choice[1]))


def visible_receiver(receiver, devices, hosts):
    return any(d["max_input_channels"] and
               d["name"].split("(")[0].strip().casefold() == receiver.casefold() and
               hosts[d["hostapi"]]["name"] != "Windows WDM-KS" for d in devices)
