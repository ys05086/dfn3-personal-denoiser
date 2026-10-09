//! The plugin window: which model is in use, the reduction limit, and the rate it runs at.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use egui::{Color32, Margin, RichText};
use nice_plug::context::gui::GuiContext;
use nice_plug::editor::dpi::LogicalSize;
use nice_plug_egui::widgets::ParamSlider;
use nice_plug_egui::{create_egui_editor, EguiEditor, EguiEditorState, EguiNiceSettings, NiceEguiApp, RepaintNotifier};
use pdn_engine::SAMPLE_RATE;

use crate::models::{self, Source};
use crate::{DenoiserParams, Shared};

const WIDTH: f32 = 460.0;
const HEIGHT: f32 = 200.0;
const MUTED: Color32 = Color32::from_rgb(150, 156, 170);
const WARNING: Color32 = Color32::from_rgb(232, 150, 90);

pub(crate) fn default_state() -> Arc<EguiEditorState> {
    EguiEditorState::from_size(LogicalSize::new(WIDTH, HEIGHT), 1.0)
}

pub(crate) struct DenoiserEditor {
    params: Arc<DenoiserParams>,
    shared: Arc<Shared>,
    gui: Option<GuiContext>,
}

pub(crate) fn create(params: Arc<DenoiserParams>, shared: Arc<Shared>, state: Arc<EguiEditorState>)
                     -> Option<EguiEditor<DenoiserEditor>> {
    create_egui_editor(
        state,
        RepaintNotifier::default(),
        EguiNiceSettings { title: "Personal Denoiser".into(), ..Default::default() },
        DenoiserEditor { params, shared, gui: None },
    )
}

impl NiceEguiApp for DenoiserEditor {
    fn build(&mut self, context: egui::Context, gui: GuiContext, _frame: &mut nice_plug_egui::Frame)
             -> Result<(), nice_plug_egui::baseview::HandlerError> {
        context.set_visuals(egui::Visuals::dark());
        self.gui = Some(gui);
        Ok(())
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut nice_plug_egui::Frame) {
        let Some(gui) = &self.gui else {
            return;
        };
        let setter = gui.param_setter();
        let status = self.shared.status.lock().map(|s| s.clone()).unwrap_or_default();
        let rate = self.shared.sample_rate.load(Ordering::Relaxed);
        egui::Frame::new().inner_margin(Margin::same(16)).show(ui, |ui| {
            ui.label(RichText::new("Personal Denoiser").size(20.0).strong());
            ui.add_space(6.0);
            let age = status.stamp.as_ref().and_then(|s| models::age(s.modified));
            let model = match status.source {
                None => "Model: loading…".to_string(),
                Some(Source::Exported) => format!("Model: your model, exported with this plugin{}",
                                                  age.map(|a| format!(" ({a})")).unwrap_or_default()),
                Some(Source::App) => format!("Model: your model from the app{}",
                                             age.map(|a| format!(", trained {a}")).unwrap_or_default()),
                Some(Source::Public) => "Model: public DeepFilterNet3 (no personal model found)".to_string(),
            };
            ui.label(model);
            if let Some(error) = &status.error {
                ui.label(RichText::new(format!("Could not use {error}")).color(WARNING).small());
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label("Max reduction");
                ui.add(ParamSlider::for_param(&self.params.reduction, &setter).with_width(240.0));
            });
            ui.add_space(10.0);
            if rate == 0 || rate as usize == SAMPLE_RATE {
                ui.label(RichText::new("48 kHz · 40 ms latency · stereo is mixed to mono").color(MUTED).small());
            } else {
                ui.label(RichText::new(format!("Bypassed: runs at 48 kHz only, the host is at {rate} Hz"))
                    .color(WARNING));
            }
        });
        // the loader may switch models at any time
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    }
}
