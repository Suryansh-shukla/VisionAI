package visionai.gui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import visionai.dao.UserDAO;
import visionai.utils.PasswordUtils;

public class ResetPasswordFrame extends JFrame {
    private final String token;
    private JPasswordField txtNewPassword, txtConfirmPassword;
    private JButton btnReset, btnCancel;
    
    public ResetPasswordFrame(String token) {
        this.token = token;
        setTitle("Reset Password");
        setSize(400, 250);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setLocationRelativeTo(null);
        
        initComponents();
    }
    
    private void initComponents() {
        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // Header
        JLabel lblTitle = new JLabel("Reset Your Password", SwingConstants.CENTER);
        lblTitle.setFont(new Font("Arial", Font.BOLD, 16));
        mainPanel.add(lblTitle, BorderLayout.NORTH);
        
        // Form Panel
        JPanel formPanel = new JPanel(new GridLayout(3, 2, 10, 10));
        
        formPanel.add(new JLabel("New Password:"));
        txtNewPassword = new JPasswordField();
        formPanel.add(txtNewPassword);
        
        formPanel.add(new JLabel("Confirm Password:"));
        txtConfirmPassword = new JPasswordField();
        formPanel.add(txtConfirmPassword);
        
        // Buttons Panel
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 10));
        btnReset = new JButton("Reset Password");
        btnCancel = new JButton("Cancel");
        
        buttonPanel.add(btnReset);
        buttonPanel.add(btnCancel);
        
        // Add action listeners
        btnReset.addActionListener(this::resetAction);
        btnCancel.addActionListener(this::cancelAction);
        
        // Add components to main panel
        mainPanel.add(formPanel, BorderLayout.CENTER);
        mainPanel.add(buttonPanel, BorderLayout.SOUTH);
        
        add(mainPanel);
    }
    
    private void resetAction(ActionEvent e) {
        String newPassword = new String(txtNewPassword.getPassword());
        String confirmPassword = new String(txtConfirmPassword.getPassword());
        
        if (newPassword.isEmpty() || confirmPassword.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please fill all fields", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        if (!newPassword.equals(confirmPassword)) {
            JOptionPane.showMessageDialog(this, "Passwords do not match", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        if (newPassword.length() < 8) {
            JOptionPane.showMessageDialog(this, "Password must be at least 8 characters", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        try {
            UserDAO userDAO = new UserDAO();
            boolean success = userDAO.resetPassword(token, newPassword);
            
            if (success) {
                JOptionPane.showMessageDialog(this, "Password reset successfully!", "Success", JOptionPane.INFORMATION_MESSAGE);
                new LoginFrame().setVisible(true);
                this.dispose();
            } else {
                JOptionPane.showMessageDialog(this, "Invalid or expired token", "Error", JOptionPane.ERROR_MESSAGE);
            }
        } catch (SQLException ex) {
            JOptionPane.showMessageDialog(this, "Database error: " + ex.getMessage(), 
                "Error", JOptionPane.ERROR_MESSAGE);
        }
    }
    
    private void cancelAction(ActionEvent e) {
        this.dispose();
    }
}