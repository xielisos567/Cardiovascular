# ================== 加载所需包（自动安装缺失包） ==================
if(!require("pacman")) install.packages("pacman")
pacman::p_load(
  xgboost, randomForest, gbm, caret, pROC, ggplot2, dplyr, tidyverse, gridExtra,
  rms, rmda, pheatmap, shapviz, reshape2,
  e1071, nnet, lightgbm, ada, glmnet
)

# 尝试安装 catboost（如果缺失）
if (!require("catboost")) {
  install.packages("catboost", repos = "https://cloud.r-project.org/")
  library(catboost)
}
catboost_available <- require(catboost)
if (!catboost_available) {
  warning("catboost could not be installed. CatBoost model will be marked as NA in outputs.")
}

# ================== 数据读取与预处理 ==================
species <- read.delim("Hypertension_marker_species.txt", row.names = 1, check.names = FALSE)
species_t <- as.data.frame(t(species))
species_t$Sample <- rownames(species_t)

sample_info <- read.delim("Hypertension_sample.txt", stringsAsFactors = FALSE)

data <- merge(species_t, sample_info, by = "Sample", all = FALSE)
data$Type <- factor(data$Type, levels = c("HC", "hypertension"))
rownames(data) <- data$Sample
data$Sample <- NULL

X <- as.matrix(data[, !names(data) %in% "Type"])
y <- data$Type
if(any(is.na(X))) X[is.na(X)] <- 0

# ================== 划分训练集和测试集 ==================
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[trainIndex, ]
X_test  <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test  <- y[-trainIndex]

y_train_num <- ifelse(y_train == "hypertension", 1, 0)
y_test_num  <- ifelse(y_test == "hypertension", 1, 0)

X_train_df <- as.data.frame(X_train)
X_test_df  <- as.data.frame(X_test)

# ================== 定义模型列表（始终包含 8 个条目） ==================
models <- list()

# 1. Logistic Regression
models$Logistic <- glm(y_train_num ~ ., data = X_train_df, family = binomial)

# 2. SVM
models$SVM <- svm(x = X_train_df, y = as.factor(y_train), probability = TRUE, kernel = "radial")

# 3. Random Forest
models$RF <- randomForest(x = X_train_df, y = as.factor(y_train), ntree = 100, importance = TRUE)

# 4. Gradient Boosting (gbm)
models$GBM <- gbm(
  formula = y_train_num ~ .,
  data = X_train_df,
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.1,
  bag.fraction = 0.5,
  verbose = FALSE
)

# 5. XGBoost
dtrain <- xgb.DMatrix(data = X_train, label = y_train_num)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test_num)
xgb_params <- list(objective = "binary:logistic", max_depth = 3, eta = 0.1)
models$XGBoost <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100, verbose = 0)

# 6. AdaBoost
models$AdaBoost <- ada(x = X_train_df, y = as.factor(y_train), iter = 100, nu = 0.1)

# 7. LightGBM
lgb_train <- lgb.Dataset(data = X_train, label = y_train_num)
lgb_params <- list(objective = "binary", metric = "binary_logloss", num_leaves = 31, learning_rate = 0.1)
models$LightGBM <- lgb.train(params = lgb_params, data = lgb_train, nrounds = 100, verbose = -1)

# 8. CatBoost（修正参数：verbose = 0）
if (catboost_available) {
  catboost_train <- catboost.load_pool(X_train, label = y_train_num)
  catboost_params <- list(iterations = 100, depth = 3, learning_rate = 0.1, loss_function = "Logloss", verbose = 0)
  models$CatBoost <- catboost.train(catboost_train, NULL, catboost_params)
} else {
  models$CatBoost <- NULL
}

# ================== 预测函数（统一输出概率） ==================
predict_prob <- function(model, newdata, type = "response") {
  if (is.null(model)) return(rep(NA, nrow(newdata)))  # 处理缺失模型
  if (inherits(model, "glm")) {
    return(predict(model, newdata, type = "response"))
  } else if (inherits(model, "svm")) {
    pred <- predict(model, newdata, probability = TRUE)
    prob_attr <- attr(pred, "probabilities")
    if ("hypertension" %in% colnames(prob_attr)) {
      return(prob_attr[, "hypertension"])
    } else {
      return(prob_attr[, 2])
    }
  } else if (inherits(model, "randomForest")) {
    return(predict(model, newdata, type = "prob")[, "hypertension"])
  } else if (inherits(model, "gbm")) {
    return(predict(model, newdata, n.trees = 100, type = "response"))
  } else if (inherits(model, "xgb.Booster")) {
    dnew <- xgb.DMatrix(data = as.matrix(newdata))
    return(predict(model, dnew))
  } else if (inherits(model, "ada")) {
    return(predict(model, newdata, type = "prob")[, 2])
  } else if (inherits(model, "lgb.Booster")) {
    return(predict(model, as.matrix(newdata)))
  } else if (inherits(model, "catboost.Model")) {
    pool <- catboost.load_pool(as.matrix(newdata))
    return(catboost.predict(model, pool, prediction_type = "Probability"))
  } else {
    stop("Unknown model type")
  }
}

# ================== 计算所有模型的训练集和测试集预测概率 ==================
train_prob <- list()
test_prob  <- list()
for (name in names(models)) {
  cat("Predicting", name, "...\n")
  if (!is.null(models[[name]])) {
    train_prob[[name]] <- predict_prob(models[[name]], X_train_df)
    test_prob[[name]]  <- predict_prob(models[[name]], X_test_df)
  } else {
    train_prob[[name]] <- rep(NA, nrow(X_train_df))
    test_prob[[name]]  <- rep(NA, nrow(X_test_df))
    cat("  Model not available, predictions set to NA.\n")
  }
}

# ================== 性能指标计算函数 ==================
calc_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
  if (all(is.na(y_pred_prob))) {
    return(c(Accuracy = NA, Sensitivity = NA, Specificity = NA,
             Precision = NA, F1 = NA, AUC = NA, Brier = NA))
  }
  y_pred <- ifelse(y_pred_prob >= threshold, 1, 0)
  y_true <- as.numeric(y_true)
  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  
  acc <- (tp + tn) / (tp + tn + fp + fn)
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  prec <- tp / (tp + fp)
  f1 <- ifelse(prec + sens == 0, 0, 2 * prec * sens / (prec + sens))
  
  roc_obj <- tryCatch(roc(y_true, y_pred_prob, quiet = TRUE), error = function(e) NULL)
  auc_val <- ifelse(is.null(roc_obj), NA, auc(roc_obj))
  brier <- mean((y_true - y_pred_prob)^2, na.rm = TRUE)
  
  return(c(Accuracy = acc, Sensitivity = sens, Specificity = spec,
           Precision = prec, F1 = f1, AUC = auc_val, Brier = brier))
}

# 计算所有模型指标
train_metrics <- list()
test_metrics  <- list()
for (name in names(models)) {
  train_metrics[[name]] <- calc_metrics(y_train_num, train_prob[[name]])
  test_metrics[[name]]  <- calc_metrics(y_test_num, test_prob[[name]])
}

train_metrics_df <- do.call(rbind, train_metrics)
test_metrics_df  <- do.call(rbind, test_metrics)
rownames(train_metrics_df) <- names(models)
rownames(test_metrics_df)  <- names(models)

# 提取训练集前5个指标（不含 AUC/Brier）
train_summary <- train_metrics_df[, 1:5]
test_summary  <- test_metrics_df[, 1:5]

# ================== 生成双层表头 CSV（确保 8 行） ==================
perf_for_csv <- data.frame(
  Train_Accuracy = train_summary[, "Accuracy"],
  Train_Sensitivity = train_summary[, "Sensitivity"],
  Train_Specificity = train_summary[, "Specificity"],
  Train_Precision = train_summary[, "Precision"],
  Train_F1 = train_summary[, "F1"],
  Test_Accuracy = test_summary[, "Accuracy"],
  Test_Sensitivity = test_summary[, "Sensitivity"],
  Test_Specificity = test_summary[, "Specificity"],
  Test_Precision = test_summary[, "Precision"],
  Test_F1 = test_summary[, "F1"]
)
rownames(perf_for_csv) <- names(models)

con <- file("model_performance_full.csv", "w")
writeLines(paste(c(rep("Train",5), rep("Test",5)), collapse="\t"), con)
writeLines(paste(c("Accuracy","Sensitivity","Specificity","Precision","F1 Score",
                   "Accuracy","Sensitivity","Specificity","Precision","F1 Score"), collapse="\t"), con)
write.table(perf_for_csv, con, sep="\t", row.names=TRUE, col.names=FALSE, quote=FALSE)
close(con)

cat("\nPerformance summary saved to model_performance_full.csv\n")
print(perf_for_csv)

# ================== ROC 曲线（过滤掉 NA 的模型） ==================
valid_models <- names(models)[!sapply(test_prob, function(x) all(is.na(x)))]
roc_list <- list()
for (name in valid_models) {
  roc_list[[name]] <- roc(y_test_num, test_prob[[name]])
}

p_roc <- ggplot() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  lapply(names(roc_list), function(name) {
    geom_step(aes(x = 1 - roc_list[[name]]$specificities,
                  y = roc_list[[name]]$sensitivities,
                  color = name), linewidth = 0.8)
  }) +
  labs(title = "ROC Curves (Test Set)", x = "1 - Specificity", y = "Sensitivity", color = "Classifier") +
  xlim(0,1) + ylim(0,1) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave("ROC_Curve.png", p_roc, width = 8, height = 6, dpi = 300)

# ================== 混淆矩阵热图 ==================
plot_confusion_matrix <- function(y_true, y_pred_prob, model_name, dataset_name, threshold = 0.5) {
  if (all(is.na(y_pred_prob))) {
    return(ggplot() + annotate("text", x=0.5, y=0.5, label="Model not available", size=5) +
           labs(title = paste(model_name, "-", dataset_name)) + theme_void())
  }
  y_pred <- ifelse(y_pred_prob >= threshold, "hypertension", "HC")
  y_true <- ifelse(y_true == 1, "hypertension", "HC")
  cm <- table(Predicted = y_pred, Actual = y_true)
  cm_df <- as.data.frame(cm)
  ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 4, color = "black") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = paste(model_name, "-", dataset_name),
         x = "True Label", y = "Predicted Label") +
    theme_minimal() +
    theme(legend.position = "none")
}

train_cm_plots <- list()
test_cm_plots  <- list()
for (name in names(models)) {
  train_cm_plots[[name]] <- plot_confusion_matrix(y_train_num, train_prob[[name]], name, "Training")
  test_cm_plots[[name]]  <- plot_confusion_matrix(y_test_num, test_prob[[name]], name, "Test")
}

n_models <- length(models)
n_cols <- 4
n_rows <- ceiling(n_models / n_cols)
train_grid <- grid.arrange(grobs = train_cm_plots, ncol = n_cols, top = "Confusion Matrices - Training Set")
test_grid  <- grid.arrange(grobs = test_cm_plots,  ncol = n_cols, top = "Confusion Matrices - Test Set")
ggsave("ConfusionMatrices_Train.png", train_grid, width = 4 * n_cols, height = 4 * n_rows, dpi = 300)
ggsave("ConfusionMatrices_Test.png", test_grid, width = 4 * n_cols, height = 4 * n_rows, dpi = 300)

# ================== 校准曲线（自定义分箱 - 修复版本） ==================
cal_curves <- function(y_true, y_pred_prob, name, n_bins = 10) {
  if (all(is.na(y_pred_prob))) return(NULL)
  
  # 计算分位数断点
  probs <- seq(0, 1, length.out = n_bins + 1)
  breaks <- quantile(y_pred_prob, probs = probs, na.rm = TRUE)
  
  # 去除重复的断点（保留唯一值）
  breaks <- unique(breaks)
  
  # 如果去重后断点少于2，无法分箱，改用等宽分箱
  if (length(breaks) < 2) {
    breaks <- seq(min(y_pred_prob, na.rm = TRUE), 
                  max(y_pred_prob, na.rm = TRUE), 
                  length.out = n_bins + 1)
    breaks <- unique(breaks)
  }
  
  # 确保至少有2个断点，否则返回空
  if (length(breaks) < 2) return(NULL)
  
  bins <- cut(y_pred_prob, breaks = breaks, include.lowest = TRUE)
  df <- data.frame(prob = y_pred_prob, true = y_true, bin = bins)
  agg <- aggregate(cbind(prob, true) ~ bin, data = df, FUN = mean)
  data.frame(mean_predicted = agg$prob, fraction_positive = agg$true, model = name)
}

cal_data <- bind_rows(lapply(names(models), function(name) {
  cal_curves(y_test_num, test_prob[[name]], name)
}))

if (!is.null(cal_data) && nrow(cal_data) > 0) {
  p_cal <- ggplot(cal_data, aes(x = mean_predicted, y = fraction_positive, color = model)) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
    labs(title = "Calibration Curves (Test Set)",
         x = "Mean Predicted Probability", y = "Fraction of Positives") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave("Calibration_curves_test.png", p_cal, width = 8, height = 6, dpi = 300)
} else {
  cat("No valid calibration data to plot.\n")
}

# ================== 决策曲线分析（取有效模型前 5 个） ==================
valid_top_models <- intersect(valid_models, names(models))[1:min(5, length(valid_models))]
if (length(valid_top_models) > 0) {
  dca_data <- data.frame(y = y_test_num)
  for (name in valid_top_models) {
    dca_data[[name]] <- test_prob[[name]]
  }
  curve_list <- list()
  for (name in valid_top_models) {
    curve_list[[name]] <- decision_curve(as.formula(paste("y ~", name)), data = dca_data,
                                          family = binomial(link = "logit"),
                                          thresholds = seq(0, 1, by = 0.01),
                                          confidence.intervals = FALSE)
  }
  png("DCA.png", width = 800, height = 600, res = 120)
  plot_decision_curve(curve_list, curve.names = valid_top_models,
                      col = rainbow(length(valid_top_models)),
                      lty = 1, lwd = 2,
                      xlab = "Threshold Probability", ylab = "Net Benefit",
                      legend.position = "topright")
  dev.off()
} else {
  cat("No valid models for DCA.\n")
}

# ================== SHAP 分析（XGBoost） ==================
if (!is.null(models$XGBoost)) {
  shp <- shapviz(models$XGBoost, X_pred = X_train, X = X_train)
  shap_summary <- sv_importance(shp, kind = "both", show_numbers = TRUE,
                                fill = "#0366d6", number_size = 3) +
                  labs(title = "SHAP Feature Importance (XGBoost)")
  ggsave("SHAP_Summary.png", shap_summary, width = 8, height = 6, dpi = 300)
} else {
  cat("XGBoost model not available, skipping SHAP plot.\n")
}

# ================== 列线图（基于重要特征构建 Logistic 模型） ==================
if (!is.null(models$XGBoost)) {
  importance <- xgb.importance(feature_names = colnames(X), model = models$XGBoost)
  top_features <- importance$Feature[1:min(5, length(importance$Feature))]
  logit_data <- data.frame(y = y_train, X_train_df[, top_features])
  logit_data$y <- as.numeric(logit_data$y == "hypertension")
  
  dd <- datadist(logit_data)
  options(datadist = "dd")
  logit_model <- lrm(y ~ ., data = logit_data, x = TRUE, y = TRUE)
  
  png("Nomogram.png", width = 1500, height = 900, res = 120)
  par(mar = c(5, 12, 4, 2))
  plot(nomogram(logit_model, fun = plogis, fun.at = c(0.1, 0.3, 0.5, 0.7, 0.9),
                funlabel = "Risk of hypertension"))
  dev.off()
} else {
  cat("XGBoost model not available, skipping nomogram.\n")
}

# ================== 森林图（单变量逻辑回归，展示全部物种） ==================
all_features <- colnames(X_train_df)
univariate_results <- data.frame()
for (feat in all_features) {
  if (sd(X_train_df[, feat]) == 0) next
  form <- as.formula(paste("y_train_num ~", feat))
  fit <- tryCatch(glm(form, data = X_train_df, family = binomial), error = function(e) NULL)
  if (is.null(fit)) next
  coef_est <- coef(fit)[2]
  se <- summary(fit)$coefficients[2, 2]
  or <- exp(coef_est)
  ci_low <- exp(coef_est - 1.96 * se)
  ci_high <- exp(coef_est + 1.96 * se)
  p_val <- summary(fit)$coefficients[2, 4]
  univariate_results <- rbind(univariate_results,
                              data.frame(Feature = feat, OR = or, CI_low = ci_low,
                                         CI_high = ci_high, p_value = p_val))
}
if (nrow(univariate_results) == 0) {
  cat("No valid features for forest plot.\n")
} else {
  univariate_results$p_adj <- p.adjust(univariate_results$p_value, method = "fdr")
  univariate_results$significance <- ifelse(univariate_results$p_adj < 0.05,
                                            "Significant (FDR < 0.05)",
                                            "Not significant")
  univariate_results <- univariate_results[order(univariate_results$OR, decreasing = TRUE), ]
  
  n_features <- nrow(univariate_results)
  plot_height <- max(6, min(40, n_features * 0.2))
  
  forest_plot <- ggplot(univariate_results, aes(x = OR, y = reorder(Feature, OR))) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey") +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2, color = "black") +
    geom_point(aes(color = significance), size = 1.5) +
    scale_color_manual(values = c("Significant (FDR < 0.05)" = "red",
                                  "Not significant" = "gray")) +
    scale_x_log10() +
    labs(title = "Forest Plot of Univariate Logistic Regression (All Features)",
         x = "Odds Ratio (95% CI)", y = "Feature",
         color = "Significance") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8),
          legend.position = "bottom")
  
  ggsave("Forest_Plot.png", forest_plot, width = 10, height = plot_height,
         dpi = 300, limitsize = FALSE)
  write.csv(univariate_results, "Forest_Plot_Data_All.csv", row.names = FALSE)
  
  cat(sprintf("Forest plot (all %d features) saved to Forest_Plot.png\n", n_features))
  cat("Data saved to Forest_Plot_Data_All.csv\n")
}

# ================== 输出测试集 AUC 和 Brier ==================
cat("\nTest AUC:\n")
for (name in names(models)) {
  if (!is.na(test_metrics[[name]]["AUC"])) {
    cat(sprintf("%s: %.3f\n", name, test_metrics[[name]]["AUC"]))
  } else {
    cat(sprintf("%s: NA\n", name))
  }
}
cat("\nTest Brier Score:\n")
for (name in names(models)) {
  if (!is.na(test_metrics[[name]]["Brier"])) {
    cat(sprintf("%s: %.4f\n", name, test_metrics[[name]]["Brier"]))
  } else {
    cat(sprintf("%s: NA\n", name))
  }
}

# ================== Bootstrap 重采样分析（随机森林） ==================
bootstrap_rf <- function(X, y, B = 1000, ntree = 100) {
  n <- nrow(X)
  auc_vec <- numeric(B)
  acc_vec <- numeric(B)
  shap_importance_list <- list()
  importance_df <- NULL   # 初始化为 NULL

  pb <- txtProgressBar(min = 0, max = B, style = 3)

  for (i in 1:B) {
    setTxtProgressBar(pb, i)

    idx_train <- sample(1:n, n, replace = TRUE)
    idx_oob <- setdiff(1:n, unique(idx_train))

    if (length(idx_oob) == 0) next

    X_train <- X[idx_train, , drop = FALSE]
    y_train <- y[idx_train]
    X_oob   <- X[idx_oob, , drop = FALSE]
    y_oob   <- y[idx_oob]

    rf_model <- randomForest(x = X_train, y = as.factor(y_train), ntree = ntree)

    pred_prob <- predict(rf_model, X_oob, type = "prob")[, "hypertension"]
    pred_class <- ifelse(pred_prob >= 0.5, "hypertension", "HC")

    roc_obj <- tryCatch(roc(y_oob, pred_prob, quiet = TRUE), error = function(e) NULL)
    if (!is.null(roc_obj)) {
      auc_vec[i] <- as.numeric(auc(roc_obj))
    } else {
      auc_vec[i] <- NA
    }
    acc_vec[i] <- mean(pred_class == y_oob)

    # SHAP 计算（可选）
    if (!is.null(rf_model)) {
      shp <- tryCatch(
        shapviz(rf_model, X_pred = X_train, X = X_train),
        error = function(e) NULL
      )
      if (!is.null(shp)) {
        shap_imp <- colMeans(abs(shp$S))
        shap_importance_list[[i]] <- shap_imp
      }
    }
  }
  close(pb)

  auc_vec <- auc_vec[!is.na(auc_vec)]
  acc_vec <- acc_vec[!is.na(acc_vec)]

  auc_mean <- mean(auc_vec)
  auc_lower <- quantile(auc_vec, 0.025, na.rm = TRUE)
  auc_upper <- quantile(auc_vec, 0.975, na.rm = TRUE)
  acc_mean <- mean(acc_vec)
  acc_lower <- quantile(acc_vec, 0.025, na.rm = TRUE)
  acc_upper <- quantile(acc_vec, 0.975, na.rm = TRUE)

  cat("\n========== Bootstrap Results (Random Forest) ==========\n")
  cat(sprintf("B = %d\n", B))
  cat(sprintf("AUC: %.3f (%.3f-%.3f)\n", auc_mean, auc_lower, auc_upper))
  cat(sprintf("ACC: %.3f (%.3f-%.3f)\n", acc_mean, acc_lower, acc_upper))

  if (length(shap_importance_list) > 0) {
    shap_matrix <- do.call(rbind, shap_importance_list)
    shap_mean <- colMeans(shap_matrix, na.rm = TRUE)
    shap_sd   <- apply(shap_matrix, 2, sd, na.rm = TRUE)
    importance_df <- data.frame(
      feature = colnames(X),
      mean_importance = shap_mean,
      sd = shap_sd
    )
    importance_df <- importance_df[order(importance_df$mean_importance, decreasing = TRUE), ]
    write.csv(importance_df, "RF_Bootstrap_Feature_Importance.csv", row.names = FALSE)
    cat("Feature importance saved to RF_Bootstrap_Feature_Importance.csv\n")
  }

  return(list(auc = auc_vec, acc = acc_vec, importance = importance_df))
}

# 运行 Bootstrap（可调整 B 值，如 100 用于快速测试）
set.seed(123)
bootstrap_results <- bootstrap_rf(X, y, B = 1000, ntree = 100)

# 绘制 AUC 和准确率分布图
if (length(bootstrap_results$auc) > 0) {
  auc_df <- data.frame(AUC = bootstrap_results$auc)
  p_auc <- ggplot(auc_df, aes(x = AUC)) + 
    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
    labs(title = "Bootstrap Distribution of AUC (RF)", x = "AUC", y = "Frequency") +
    theme_minimal()
  ggsave("Bootstrap_AUC_distribution.png", p_auc, width = 8, height = 6, dpi = 300)
  
  acc_df <- data.frame(Accuracy = bootstrap_results$acc)
  p_acc <- ggplot(acc_df, aes(x = Accuracy)) + 
    geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
    labs(title = "Bootstrap Distribution of Accuracy (RF)", x = "Accuracy", y = "Frequency") +
    theme_minimal()
  ggsave("Bootstrap_Accuracy_distribution.png", p_acc, width = 8, height = 6, dpi = 300)
}

# ================== 各模型 AUC 值统计表 ==================
auc_table <- data.frame(
  Model = names(models),
  Train_AUC = sapply(names(models), function(m) {
    auc_val <- ifelse(!is.null(train_metrics[[m]]) && length(train_metrics[[m]]) >= 7,
                      train_metrics[[m]]["AUC"], NA)
    if (!is.na(auc_val)) round(auc_val, 3) else NA
  }),
  Test_AUC = sapply(names(models), function(m) {
    auc_val <- ifelse(!is.null(test_metrics[[m]]) && length(test_metrics[[m]]) >= 7,
                      test_metrics[[m]]["AUC"], NA)
    if (!is.na(auc_val)) round(auc_val, 3) else NA
  })
)

# 保存为 CSV
write.csv(auc_table, "model_AUC_table.csv", row.names = FALSE)

# 打印表格
cat("\n========== Model AUC Summary ==========\n")
print(auc_table)
