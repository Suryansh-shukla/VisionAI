package visionai.gui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import visionai.dao.UserDAO;

public class ForgotPasswordFrame extends JFrame {
    private JTextField txtEmail;
    private JButton btnSubmit, btnBack;
    
    public ForgotPasswordFrame() {
        setTitle("Password Recovery");
        setSize(400, 200);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setLocationRelativeTo(null);
        
        initComponents();
    }
    
    private void initComponents() {
        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // Header
        JLabel lblTitle = new JLabel("Password Recovery", SwingConstants.CENTER);
        lblTitle.setFont(new Font("Arial", Font.BOLD, 16));
        mainPanel.add(lblTitle, BorderLayout.NORTH);
        
        // Form Panel
        JPanel formPanel = new JPanel(new GridLayout(2, 2, 10, 10));
        
        formPanel.add(new JLabel("Email Address:"));
        txtEmail = new JTextField();
        formPanel.add(txtEmail);
        
        // Buttons Panel
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 10));
        btnSubmit = new JButton("Submit");
        btnBack = new JButton("Back to Login");
        
        buttonPanel.add(btnSubmit);
        buttonPanel.add(btnBack);
        
        // Add action listeners
        btnSubmit.addActionListener(this::submitAction);
        btnBack.addActionListener(this::backAction);
        
        // Add components to main panel
        mainPanel.add(formPanel, BorderLayout.CENTER);
        mainPanel.add(buttonPanel, BorderLayout.SOUTH);
        
        add(mainPanel);
    }
    
    private void submitAction(ActionEvent e) {
        String email = txtEmail.getText().trim();
        
        if (email.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter your email address", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        try {
            UserDAO userDAO = new UserDAO();
            String token = userDAO.createPasswordResetToken(email);
            
            if (token != null) {
                // In a real application, you would send this token via email
                JOptionPane.showMessageDialog(this, 
                    "Password reset link has been sent to your email.\n" +
                    "For demo purposes, your token is: " + token, 
                    "Reset Link Sent", JOptionPane.INFORMATION_MESSAGE);
                
                new ResetPasswordFrame(token).setVisible(true);
                this.dispose();
            } else {
                JOptionPane.showMessageDialog(this, "Email address not found", "Error", JOptionPane.ERROR_MESSAGE);
            }
        } catch (SQLException ex) {
            JOptionPane.showMessageDialog(this, "Database error: " + ex.getMessage(), 
                "Error", JOptionPane.ERROR_MESSAGE);
        }
    }
    
    private void backAction(ActionEvent e) {
        this.dispose();
    }
}